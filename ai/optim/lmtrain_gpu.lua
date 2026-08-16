--  The CPU path is never touched.  If backend ~= "gpu" none of this runs.

local lmlog = require("lmlog")

local G = {}

local floor, ceil, huge, sqrt = math.floor, math.ceil, math.huge, math.sqrt
local sformat = string.format


--  lmio (:save/:load), LMTrain._snapshot and LMTrain._restore all walk
--  `params[i].data.data` as a flat 1-indexed Lua table.  We hand them a
--  host mirror of each GPU parameter with exactly that shape, so every
--  existing persistence path keeps working verbatim. The mirror is
--  pulled from VRAM on demand and pushed back when marked dirty.
local function make_shims(model)
    local shims = {}
    for i = 1, #model.params do
        local p = model.params[i]
        shims[i] = {
            name = p.name,
            is_param = true,
            gpu = p,
            data = { rows = p.rows, cols = p.cols, data = {} },
            grad = { rows = p.rows, cols = p.cols, data = {} },
        }
    end
    return shims
end

function G.install(LMTrain)
    if LMTrain._gpu_installed then return LMTrain end
    LMTrain._gpu_installed = true

    --  Host <-> device weight mirroring
    function LMTrain:_gpu_pull()
        if not self.gpu_model then return self end
        for i = 1, #self.params do
            local s = self.params[i]
            s.data.data = s.gpu.w:toflat()
        end
        self._host_dirty = false
        return self
    end

    function LMTrain:_gpu_push()
        if not self.gpu_model then return self end
        for i = 1, #self.params do
            local s = self.params[i]
            local flat = s.data.data
            if flat and #flat == s.gpu.nelem then s.gpu.w:upload(flat) end
        end
        self._host_dirty = false
        return self
    end

    --  Build
    function LMTrain:_build_gpu()
        local ok_ad, adapter = pcall(require, "luatl_adapter")
        if not ok_ad or not adapter.available then
            error(sformat(
                "LMTrain: backend = 'gpu' was requested but luaTL is unavailable: %s\n" ..
                "  Build the CUDA backend and put luaTL.so / luaTL.dll on the load path:\n" ..
                "    nvcc -O3 -std=c++14 --shared -Xcompiler -fPIC \\\n" ..
                "         -gencode arch=compute_86,code=sm_86 --use_fast_math \\\n" ..
                "         luaTL_train.cu -o luaTL.so\n" ..
                "  Or set backend = 'cpu' to use the pure-Lua autograd path.",
                ok_ad and tostring(adapter.error or "not available")
                       or tostring(adapter)), 3)
        end

        local ok_gt, GT = pcall(require, "gpu_transformer")
        if not ok_gt then
            error("LMTrain: backend = 'gpu' but gpu_transformer.lua failed to load: "
                  .. tostring(GT), 3)
        end

        local bs = self.batch_size
        local ok_m, model = pcall(GT.new, {
            vocab = self.vocab, dim = self.dim, layers = self.layers,
            heads = self.heads, ffn = self.ffn, max_seq = self.max_seq,
            batch_size = bs, causal = self.causal, pos = self.pos,
            rope_base = self.rope_base, tie = self.tie, dropout = self.dropout,
            clip_norm = self.clip_norm, weight_decay = self.weight_decay,
            beta1 = self.beta1, beta2 = self.beta2,
            graph = self.graph, timing = true,
            want_acc = true,
        })
        if not ok_m then
            error("LMTrain: GPU model construction failed: " .. tostring(model), 3)
        end

        local opt = string.lower(tostring(self.optimizer or "adamw"))
        if opt ~= "adamw" and opt ~= "adam" then
            error(sformat("LMTrain: backend = 'gpu' implements AdamW only, got optimizer = '%s'.\n" ..
                  "  Use optimizer = 'adamw', or backend = 'cpu' for %s.", opt, opt), 3)
        end

        self.gpu_model = model
        self.model     = model
        self.params    = make_shims(model)
        self.opt       = { lr = self.lr, zero_grad = function() end,
                           step = function() end, backend = "gpu" }
        self.sgd       = self.opt
        self.base_lr   = self.lr
        self.best_snapshot = nil
        self._dirty      = false
        self._host_dirty = false

        if self.verbose then
            local hw = adapter.hardware()
            local gf = model:flops_per_step()
            print(sformat(
                "[LMTrain] GPU backend ready: %s (%s, %d SMs, %.1f TFLOP/s fp32 peak)",
                hw.name, hw.sm, hw.multiprocessors, hw.gflops_fp32 / 1000))
            print(sformat(
                "[LMTrain] %s params | %d x %d tokens/step | %.2f GFLOP/step | tie=%s",
                lmlog.fmt_count(model:nparams()), model.B, model.T,
                gf / 1e9, tostring(self.tie)))
        end
        return self
    end

    --  evaluation (forward only)
    function LMTrain:_evaluate_gpu(windows)
        local m  = self.gpu_model
        local bs = m.B
        local tot, cnt, cor = 0, 0, 0
        local i = 1
        while i <= #windows do
            local batch, n = {}, 0
            while n < bs and i <= #windows do
                n = n + 1; batch[n] = windows[i]; i = i + 1
            end
            local loss, correct, valid = m:eval_step(batch, n)
            tot = tot + loss * valid
            cnt = cnt + valid
            cor = cor + correct
        end
        if cnt == 0 then return 0, 0 end
        return tot / cnt, 100 * cor / cnt
    end

    --  The GPU run loop
    function LMTrain:_run_gpu(train_w, val_w, build_windows)
        local m  = self.gpu_model
        local bs = m.B
        local nwin = #train_w
        local nbatch = ceil(nwin / bs)
        local total_steps = self.epochs * nbatch
        local order = {}
        for i = 1, nwin do order[i] = i end

        if self._host_dirty then self:_gpu_push() end

        local fpstep = m:flops_per_step()

        local rep = lmlog.new{
            total_steps = total_steps,
            bar_style   = self.bar_style,
            bar_len     = self.bar_len,
            interval    = self.log_every_sec or 0.2,
            inplace     = self.inplace_bar,
        }
        self._reporter = rep

        local gstep, stalled = 0, 0
        local batch = {}
        self.stopped = nil

        for epoch = 1, self.epochs do
            self.epoch = epoch

            if self.shuffle then
                for i = nwin, 2, -1 do
                    local j = math.random(i)
                    order[i], order[j] = order[j], order[i]
                end
            end

            local sum_loss, sum_tok, correct, skipped = 0, 0, 0, 0
            local last_norm, ep_gpu_ms = 0, 0

            for b = 1, nbatch do
                local lo = (b - 1) * bs + 1
                local hi = lo + bs - 1
                if hi > nwin then hi = nwin end
                local n = hi - lo + 1
                for k = 1, n do batch[k] = train_w[order[lo + k - 1]] end

                gstep = gstep + 1
                local lr = self:_lr_at(gstep, total_steps)
                self.cur_lr = lr
                self.opt.lr = lr

                -- token ids in -> loss out -> weights updated
                local loss, corr, tok, gnorm, finite =
                    m:train_step(batch, n, lr, (m.steps or 0) + 1)

                if not finite then skipped = skipped + 1 else
                    sum_loss = sum_loss + loss * tok
                    sum_tok  = sum_tok + tok
                    correct  = correct + corr
                    last_norm = gnorm
                end
                ep_gpu_ms = ep_gpu_ms + (m:last_step_ms() or 0)
                rep:tick(tok)

                if self.verbose then
                    rep:update({
                        step = gstep, epoch = epoch, epochs = self.epochs,
                        batch = b, nbatch = nbatch,
                        loss = loss, lr = lr, gnorm = gnorm,
                        acc = tok > 0 and (100 * corr / tok) or nil,
                        tflops = (m:last_step_ms() or 0) > 0
                                 and (fpstep / (m:last_step_ms() * 1e-3) / 1e12) or nil,
                        val_loss = self.val_loss,
                    })
                end

                m:maybe_capture()
            end

            local avg = sum_tok > 0 and (sum_loss / sum_tok) or huge
            self.loss      = avg
            self.accuracy  = sum_tok > 0 and (100 * correct / sum_tok) or 0
            self.grad_norm = last_norm
            self.elapsed   = rep:elapsed()
            self.tokens_per_sec = rep.rate
            self.gpu_ms_per_epoch = ep_gpu_ms
            self.tflops = ep_gpu_ms > 0
                and (fpstep * nbatch / (ep_gpu_ms * 1e-3) / 1e12) or 0

            if #val_w > 0 and (epoch % self.eval_every == 0 or epoch == self.epochs) then
                self.val_loss, self.val_accuracy = self:_evaluate_gpu(val_w)
            end

            local monitor = self.val_loss or avg

            self.history[#self.history + 1] = {
                epoch = epoch, loss = avg, val_loss = self.val_loss,
                accuracy = self.accuracy, lr = self.cur_lr,
                grad_norm = last_norm, time = self.elapsed,
                skipped = skipped, tflops = self.tflops,
            }

            if avg ~= avg or avg == huge or avg == -huge then
                self.stopped = "diverged"
                self.is_best = false
                rep:say(sformat("LMTrain: stopping at epoch %d, loss became %s. " ..
                    "No update was applied with non-finite gradients; try a smaller lr.",
                    epoch, tostring(avg)))
                break
            end

            self.is_best = monitor < (self.best_loss - self.min_delta)
            if self.is_best then
                self.best_loss = monitor
                stalled = 0
                if self.keep_best then
                    -- FIX: Use GPU-to-GPU snapshot to avoid CPU RAM OOM!
                    self.best_snapshot = m:snapshot()
                end
            else
                if monitor < self.best_loss then self.best_loss = monitor end
                stalled = stalled + 1
            end

            if self.verbose and (epoch % self.every == 0
                or epoch == self.epochs or epoch == 1) then
                if type(self.log) == "function" then
                    local ok, err = pcall(self.log, self)
                    if not ok then
                        error(sformat("LMTrain: custom log function failed at epoch %d: %s",
                              epoch, tostring(err)), 2)
                    end
                else
                    rep:update({
                        step = gstep, epoch = epoch, epochs = self.epochs,
                        batch = nbatch, nbatch = nbatch,
                        loss = avg, acc = self.accuracy, lr = self.cur_lr,
                        gnorm = last_norm, val_loss = self.val_loss,
                        best = self.is_best, tflops = self.tflops,
                    }, true)
                end
            end

            if self.stop_loss and monitor <= self.stop_loss then
                self.stopped = "stop_loss"
                rep:say(sformat("Stopped early at epoch %d: loss %.5f reached stop_loss %.5f",
                    epoch, monitor, self.stop_loss))
                break
            end

            if self.stop_when then
                local ok, hit = pcall(self.stop_when, self)
                if not ok then
                    error(sformat("LMTrain: stop_when failed at epoch %d: %s",
                          epoch, tostring(hit)), 2)
                end
                if hit then
                    self.stopped = "stop_when"
                    rep:say(sformat("Stopped early at epoch %d: stop_when condition met", epoch))
                    break
                end
            end

            if self.patience > 0 and stalled >= self.patience then
                self.stopped = "patience"
                rep:say(sformat("Stopped early at epoch %d: no improvement for %d epochs (best %.5f)",
                    epoch, stalled, self.best_loss))
                break
            end
            collectgarbage("collect")
        end

        if self.stopped == nil then self.stopped = "epochs" end
        rep:finish()

        if self.keep_best and self.best_snapshot and self.stopped ~= "epochs" then
            if m:restore(self.best_snapshot) and self.verbose then
                print(sformat("Restored best weights (loss %.5f)", self.best_loss))
            end
        end

        self:_gpu_pull()
        self.opt.lr = self.base_lr
        self.lr = self.base_lr

        if type(self.on_stop) == "function" then pcall(self.on_stop, self) end
        return self.model
    end

    --  :load()
    local prev_load = LMTrain.load
    if type(prev_load) == "function" then
        function LMTrain:load(...)
            local r = { prev_load(self, ...) }
            if self.gpu_model then
                self._host_dirty = true
                self:_gpu_push()
            end
            return unpack and unpack(r) or table.unpack(r)
        end
    end

    return LMTrain
end

return G
