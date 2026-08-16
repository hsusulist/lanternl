--  The CPU path is never touched.  If backend ~= "gpu" none of this runs.

local lmlog = require("lmlog")

local G = {}

local ceil, huge = math.ceil, math.huge
local sformat = string.format

--  lmio (:save/:load) and LMTrain._snapshot/_restore walk
--  `params[i].data.data` as a flat 1-indexed Lua table.  We hand them a
--  host mirror of each GPU parameter with exactly that shape so the
--  metadata (rows/cols/count) is right on both backends.
--
--  The `.data` array itself stays EMPTY by default.  Materialising it for a
--  134M-parameter model means a 134M-slot Lua table (>1 GB in LuaJIT, plus
--  GC churn) for a mirror nothing reads.  lmio instead goes through
--  _param_get/_param_set, overridden below to stream fixed-size chunks
--  straight out of VRAM.  _gpu_pull() remains available for code that
--  genuinely wants the host copy, and now says no when that would be
--  ruinous unless explicitly forced.
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

    local cpu_param_get   = LMTrain._param_get
    local cpu_param_set   = LMTrain._param_set
    local cpu_param_dtype = LMTrain._param_dtype

    -- ---- lmio streaming hooks ---------------------------------------
    function LMTrain:_param_dtype()
        if self.gpu_model then return "f32" end
        return cpu_param_dtype and cpu_param_dtype(self) or "f64"
    end

    function LMTrain:_param_get(i, from, count, dst)
        if not self.gpu_model then return cpu_param_get(self, i, from, count, dst) end
        return self.gpu_model:read_param(i, from, count, dst)
    end

    function LMTrain:_param_set(i, from, count, src)
        if not self.gpu_model then return cpu_param_set(self, i, from, count, src) end
        return self.gpu_model:write_param(i, from, count, src)
    end

    --  Host <-> device weight mirroring (explicit, no longer automatic)
    local PULL_WARN = 8e6      -- elements

    function LMTrain:_gpu_pull(force)
        if not self.gpu_model then return self end
        local n = self.gpu_model:nparams()
        if n > PULL_WARN and not force then
            error(sformat("LMTrain:_gpu_pull: refusing to materialise %s parameters as Lua "
                .. "tables (~%s of host RAM). Use train:save(path) -- it streams straight "
                .. "from VRAM -- or pass force = true if you really need the mirror.",
                lmlog.fmt_count(n), lmlog.fmt_bytes(n * 16)), 2)
        end
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

        -- Release the previous device model BEFORE allocating the next one.
        -- :load() and any ARCH_KEYS change route through _build(), so this
        -- used to leave a full parameter + activation set alive until GC ran
        -- the handle finalizers -- i.e. train:load() twice was ~2x VRAM.
        -- best_snapshot must be dropped first: it aliases handles that
        -- Model:free() is about to release, and restoring it into a NEW model
        -- would hand freed device pointers to luaTL_gpu_copy.
        if self.gpu_model then
            if self.best_snapshot then
                pcall(function() self.gpu_model:release_snapshot(self.best_snapshot) end)
                self.best_snapshot = nil
            end
            local old = self.gpu_model
            self.gpu_model, self.model, self.params = nil, nil, {}
            pcall(function() old:free() end)
            collectgarbage("collect")
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
            error(sformat("LMTrain: backend = 'gpu' implements AdamW only, got optimizer = '%s'.\n"
                .. "  Use optimizer = 'adamw', or backend = 'cpu' for %s.", opt, opt), 3)
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
            lmlog.say(sformat(
                "[LMTrain] GPU backend ready: %s (%s, %d SMs, %.1f TFLOP/s fp32 peak)",
                hw.name, hw.sm, hw.multiprocessors, hw.gflops_fp32 / 1000))
            lmlog.say(sformat(
                "[LMTrain] %s params | %d x %d tokens/step | %.2f GFLOP/step | tie=%s",
                lmlog.fmt_count(model:nparams()), model.B, model.T,
                gf / 1e9, tostring(self.tie)))
            -- PROBLEM 2, GPU side: quantify what an oversized vocab costs.
            local vram_v = self.vocab * self.dim * 4 * (self.tie and 4 or 8)
            local logits = model.B * model.T * self.vocab * 4
            lmlog.say(sformat(
                "[LMTrain] vocab = %d costs ~%s of embedding state + ~%s per logits buffer",
                self.vocab, lmlog.fmt_bytes(vram_v), lmlog.fmt_bytes(logits)))
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
        -- eval_step moves the model's _valid scalar, which is what gates CUDA
        -- graph replay.  Nothing to undo (train_step re-sets it every step),
        -- but worth stating: replay simply falls back to a queued run for one
        -- step after validation, which is correct, just marginally slower.
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
            interval    = self.log_every_sec,
            inplace     = self.inplace_bar,
        }
        self._reporter = rep

        local gstep, stalled = 0, 0
        local batch = {}
        self.stopped = nil

        self:_autosave_begin()

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
                for k = n + 1, #batch do batch[k] = nil end   -- no stale windows

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

                -- Safe between steps: the checkpoint only reads p.w.t via
                -- luaTL_download_f32 and never touches _valid, the CE/sum/adam
                -- command structs, or the captured graph.
                if self._as and self._as.seconds then self:_autosave_tick("batch") end
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

            -- same stale-validation gate as the CPU loop
            local did_eval = false
            if #val_w > 0 and (epoch % self.eval_every == 0 or epoch == self.epochs) then
                self.val_loss, self.val_accuracy = self:_evaluate_gpu(val_w)
                did_eval = true
            end
            local use_val = (#val_w > 0)
            self._monitor_valid = (not use_val) or did_eval
            local monitor = use_val and self.val_loss or avg

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

            if self._monitor_valid then
                self.is_best = monitor < (self.best_loss - self.min_delta)
                if self.is_best then
                    self.best_loss = monitor
                    stalled = 0
                    if self.keep_best then
                        -- GPU-to-GPU snapshot: no CPU RAM used.  Now writes
                        -- INTO a reusable buffer -- the old version allocated a
                        -- complete fresh parameter set in VRAM on every
                        -- improvement and relied on GC finalizers to reclaim
                        -- the previous one, so several full copies could be
                        -- live at once on exactly the OOM-prone hardware this
                        -- mechanism was added to protect.
                        self.best_snapshot = m:snapshot(self.best_snapshot)
                    end
                else
                    if monitor < self.best_loss then self.best_loss = monitor end
                    stalled = stalled + 1
                end
            else
                self.is_best = false
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

            self:_autosave_tick("epoch")

            if self.stop_loss and self._monitor_valid and monitor <= self.stop_loss then
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
                rep:say(sformat("Stopped early at epoch %d: no improvement for %d epochs "
                    .. "(best %.5f)", epoch, stalled, self.best_loss))
                break
            end
            collectgarbage("collect")
        end

        if self.stopped == nil then self.stopped = "epochs" end
        rep:finish()

        if self:_should_restore_best() then
            if m:restore(self.best_snapshot) and self.verbose then
                rep:say(sformat("Restored best weights (loss %.5f)", self.best_loss))
            end
        end

        -- Final checkpoint AFTER the restore, so the file matches the weights
        -- the caller is left with.  Streams from VRAM; no host mirror needed.
        self:_autosave_tick("final")
        self:_autosave_end()

        -- The unconditional self:_gpu_pull() that used to live here built a
        -- Lua table per parameter element at the end of EVERY run -- >1 GB of
        -- host RAM for a 134M model, for a mirror nothing reads.  Persistence
        -- now streams from VRAM instead.  Small models still get the mirror
        -- for backwards compatibility with code that pokes params[i].data.
        if m:nparams() <= PULL_WARN then
            pcall(function() self:_gpu_pull() end)
        else
            self._host_dirty = false
            self.params_on_device = true
        end

        self.opt.lr = self.base_lr
        self.lr = self.base_lr
        self._reporter = nil

        if type(self.on_stop) == "function" then pcall(self.on_stop, self) end
        return self.model
    end

    --  :load()
    local prev_load = LMTrain.load
    if type(prev_load) == "function" then
        function LMTrain:load(...)
            local r = { prev_load(self, ...) }
            -- lmio's :load() writes through _param_set, which on the GPU path
            -- uploads directly to VRAM -- no host mirror to push afterwards.
            if self.gpu_model then self._host_dirty = false end
            -- was: `return unpack and unpack(r) or table.unpack(r)`, which
            -- evaluated table.unpack (nil in LuaJIT 5.1) whenever the wrapped
            -- call returned nil/false.
            local un = unpack or table.unpack
            return un(r, 1, select("#", ...) and #r or #r)
        end
    end

    return LMTrain
end

return G
