/* SingleHDR Frontend */

const $ = (sel) => document.querySelector(sel);
const show = (el) => el.classList.remove('hidden');
const hide = (el) => el.classList.add('hidden');

let currentJobId = null;
let evtSource = null;

// Client-side HDR data for tone mapping
let hdrData = null; // { width, height, pixels: Float32Array }
let inputLocalUrl = null;
let inputLinearMetrics = null;

// Batch state
let batchQueue = []; // [{ file, jobId, status, data, errorMsg }]
let batchIndex = -1;
let batchProcessing = false;

// --- Drop Zone ---
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFiles(Array.from(fileInput.files));
    fileInput.value = '';
});

// --- File Handling ---
async function handleFiles(files) {
    const validFiles = files.filter(f => f.type.match(/^image\/(png|jpeg|jpg)$/));
    if (!validFiles.length) {
        showError('Please upload PNG or JPEG images.');
        return;
    }

    // Reset UI
    hide($('#result-section'));
    hide($('#error-section'));
    hide($('#progress-section'));
    hdrData = null;
    stopProgress();

    if (validFiles.length === 1) {
        // Single file mode
        hide($('#batch-section'));
        batchQueue = [];
        batchIndex = -1;
        batchProcessing = false;
        await uploadSingleFile(validFiles[0]);
    } else {
        // Batch mode — upload all first, then wait for Generate
        batchQueue = validFiles.map(f => ({ file: f, jobId: null, status: 'uploading', data: null }));
        batchIndex = -1;
        batchProcessing = false;
        show($('#batch-section'));
        renderBatchList();
        await uploadBatchFiles();
    }
}

// --- Single File Upload ---
async function uploadSingleFile(file) {
    // Show local preview
    if (inputLocalUrl) URL.revokeObjectURL(inputLocalUrl);
    inputLocalUrl = URL.createObjectURL(file);
    $('#input-preview').src = inputLocalUrl;

    const fd = new FormData();
    fd.append('file', file);

    try {
        const resp = await fetch('/api/upload', { method: 'POST', body: fd });
        if (!resp.ok) {
            const err = await resp.json();
            showError(err.detail || err.error || 'Upload failed');
            return;
        }
        const data = await resp.json();
        currentJobId = data.job_id;
        displayInputInfo(data);
        show($('#input-info'));
        show($('#controls'));
    } catch (e) {
        showError('Upload failed: ' + e.message);
    }
}

// --- Batch Upload ---
async function uploadBatchFiles() {
    for (let i = 0; i < batchQueue.length; i++) {
        const item = batchQueue[i];
        if (item.status !== 'uploading') continue;

        const fd = new FormData();
        fd.append('file', item.file);

        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: fd });
            if (!resp.ok) {
                const err = await resp.json();
                item.status = 'error';
                item.errorMsg = err.detail || err.error || 'Upload failed';
            } else {
                const data = await resp.json();
                item.jobId = data.job_id;
                item.status = 'uploaded';
                item.data = { uploadData: data };
            }
        } catch (e) {
            item.status = 'error';
            item.errorMsg = 'Upload failed';
        }
        renderBatchList();
    }

    // Show first uploaded image info + controls
    const firstIdx = batchQueue.findIndex(item => item.status === 'uploaded');
    if (firstIdx !== -1) {
        batchIndex = firstIdx;
        showBatchItem(firstIdx);
        show($('#controls'));
    }
}

// --- Batch List Rendering ---
function renderBatchList() {
    const list = $('#batch-list');
    list.innerHTML = batchQueue.map((item, i) => {
        const removable = item.status === 'uploaded' && !batchProcessing;
        return `
        <div class="batch-item ${i === batchIndex ? 'active' : ''}" data-idx="${i}">
            <span class="batch-item-name">${item.file.name}</span>
            <span class="batch-item-status ${item.status}" ${item.errorMsg ? `title="${item.errorMsg}"` : ''}>${item.status === 'uploaded' ? 'ready' : item.status}</span>
            ${removable ? `<button class="batch-remove-btn" data-idx="${i}" title="Remove">&times;</button>` : ''}
        </div>`;
    }).join('');

    // Click to view
    list.querySelectorAll('.batch-item').forEach(el => {
        el.addEventListener('click', (e) => {
            if (e.target.classList.contains('batch-remove-btn')) return;
            const idx = parseInt(el.dataset.idx);
            const st = batchQueue[idx].status;
            if (st === 'uploaded' || st === 'complete') {
                batchIndex = idx;
                showBatchItem(idx);
                renderBatchList();
            }
        });
    });

    // Remove buttons
    list.querySelectorAll('.batch-remove-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const idx = parseInt(btn.dataset.idx);
            batchQueue.splice(idx, 1);
            if (batchQueue.length === 0) {
                hide($('#batch-section'));
                hide($('#controls'));
                hide($('#input-info'));
            } else {
                if (batchIndex >= batchQueue.length) batchIndex = batchQueue.length - 1;
                else if (batchIndex > idx) batchIndex--;
                renderBatchList();
            }
        });
    });
}

function showBatchItem(idx) {
    const item = batchQueue[idx];
    if (!item) return;

    currentJobId = item.jobId;
    if (inputLocalUrl) URL.revokeObjectURL(inputLocalUrl);
    inputLocalUrl = URL.createObjectURL(item.file);
    $('#input-preview').src = inputLocalUrl;

    if (item.data && item.data.uploadData) {
        displayInputInfo(item.data.uploadData);
        show($('#input-info'));
    }

    if (item.status === 'complete' && item.data && item.data.resultData) {
        displayResult(item.data.resultData);
        loadHdrData(item.jobId);
    } else {
        hide($('#result-section'));
        hdrData = null;
    }
}

function displayInputInfo(data) {
    $('#input-dims').textContent = `${data.width} x ${data.height}`;
    $('#input-size').textContent = formatBytes(data.file_size_bytes);
    $('#input-format').textContent = data.format;
    $('#input-dr').textContent = `${data.dynamic_range_ev.toFixed(1)} EV`;
    $('#input-brightness').textContent = `${data.mean_brightness.toFixed(0)} / 255`;
    $('#input-clipping').textContent = `${data.clipping_percent.toFixed(1)}%`;

    // Linear-domain metrics (same space as HDR output)
    $('#input-mean-linear').textContent = data.mean_luminance_linear.toFixed(4);
    $('#input-peak-linear').textContent = data.peak_luminance_linear.toFixed(4);

    // Store for comparison with HDR result
    inputLinearMetrics = {
        mean: data.mean_luminance_linear,
        peak: data.peak_luminance_linear,
        contrast: data.contrast_ratio,
    };

    drawHistogram($('#histogram-canvas'), data.histogram);
}

// --- Slider labels ---
$('#thr-slider').addEventListener('input', (e) => {
    $('#thr-val').textContent = parseFloat(e.target.value).toFixed(2);
});
$('#exposure-slider').addEventListener('input', (e) => {
    $('#exposure-val').textContent = parseFloat(e.target.value).toFixed(1);
    applyClientTonemap();
});
$('#gamma-slider').addEventListener('input', (e) => {
    $('#gamma-val').textContent = parseFloat(e.target.value).toFixed(1);
    applyClientTonemap();
});
$('#tonemap-select').addEventListener('change', () => {
    applyClientTonemap();
});

// --- Generate ---
$('#generate-btn').addEventListener('click', async () => {
    if (batchQueue.length > 0) {
        // Batch mode — process all uploaded items
        const hasUploaded = batchQueue.some(item => item.status === 'uploaded');
        if (!hasUploaded) return;
        batchProcessing = true;
        $('#generate-btn').disabled = true;
        renderBatchList();
        await processBatchNext();
        return;
    }

    // Single mode
    if (!currentJobId) return;

    const mode = document.querySelector('input[name="mode"]:checked').value;
    const hallucination_threshold = parseFloat($('#thr-slider').value);
    hide($('#error-section'));
    hide($('#result-section'));
    hdrData = null;

    $('#generate-btn').disabled = true;

    try {
        const resp = await fetch(`/api/generate/${currentJobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, hallucination_threshold }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            showError(err.detail || 'Generation failed');
            $('#generate-btn').disabled = false;
            return;
        }

        show($('#progress-section'));
        startSSE(currentJobId);
    } catch (e) {
        showError('Request failed: ' + e.message);
        $('#generate-btn').disabled = false;
    }
});

// --- Batch Processing ---
async function processBatchNext() {
    const nextIdx = batchQueue.findIndex(item => item.status === 'uploaded');
    if (nextIdx === -1) {
        // All done
        batchProcessing = false;
        $('#generate-btn').disabled = false;
        renderBatchList();
        return;
    }

    batchIndex = nextIdx;
    batchQueue[nextIdx].status = 'processing';
    currentJobId = batchQueue[nextIdx].jobId;
    showBatchItem(nextIdx);
    renderBatchList();

    const mode = document.querySelector('input[name="mode"]:checked').value;
    const hallucination_threshold = parseFloat($('#thr-slider').value);
    const jobId = currentJobId;

    try {
        const resp = await fetch(`/api/generate/${jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode, hallucination_threshold }),
        });

        if (!resp.ok) {
            batchQueue[nextIdx].status = 'error';
            renderBatchList();
            await processBatchNext();
            return;
        }

        show($('#progress-section'));
        const result = await waitForCompletion(jobId);

        if (result.ok) {
            try {
                const resultResp = await fetch(`/api/result/${jobId}`);
                if (resultResp.ok) {
                    const resultData = await resultResp.json();
                    batchQueue[nextIdx].status = 'complete';
                    batchQueue[nextIdx].data.resultData = resultData;
                    renderBatchList();
                    if (batchIndex === nextIdx) {
                        displayResult(resultData);
                        await loadHdrData(jobId);
                    }
                } else {
                    batchQueue[nextIdx].status = 'error';
                    batchQueue[nextIdx].errorMsg = 'Failed to load result';
                    showError('Failed to load result');
                    renderBatchList();
                }
            } catch (e) {
                batchQueue[nextIdx].status = 'error';
                batchQueue[nextIdx].errorMsg = e.message;
                renderBatchList();
            }
        } else {
            batchQueue[nextIdx].status = 'error';
            batchQueue[nextIdx].errorMsg = result.message;
            showError(result.message);
            renderBatchList();
        }
    } catch (e) {
        batchQueue[nextIdx].status = 'error';
        renderBatchList();
    }

    hide($('#progress-section'));
    await processBatchNext();
}

function waitForCompletion(jobId) {
    return new Promise((resolve) => {
        let done = false;
        let lastMessage = '';
        let sseGotMessage = false;
        let batchPollTimer = null;
        const es = new EventSource(`/api/status/${jobId}`);

        function finish(result) {
            if (done) return;
            done = true;
            es.close();
            if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null; }
            resolve(result);
        }

        function handleData(data) {
            updateProgress(data);
            lastMessage = data.message || '';
            if (data.stage === 'complete') finish({ ok: true });
            else if (data.stage === 'error') finish({ ok: false, message: data.message });
            else if (data.stage === 'cancelled') finish({ ok: false, message: 'Cancelled' });
        }

        function startBatchPolling() {
            if (batchPollTimer) return;
            batchPollTimer = setInterval(async () => {
                try {
                    const resp = await fetch(`/api/status-poll/${jobId}`);
                    if (!resp.ok) return;
                    handleData(await resp.json());
                } catch (e) { /* retry next tick */ }
            }, 1000);
        }

        es.onmessage = (e) => {
            sseGotMessage = true;
            handleData(JSON.parse(e.data));
        };
        es.onerror = () => {
            es.close();
            if (!sseGotMessage) startBatchPolling();
            else finish({ ok: false, message: lastMessage || 'Server connection lost' });
        };

        setTimeout(() => {
            if (!sseGotMessage && !done) {
                es.close();
                startBatchPolling();
            }
        }, 3000);
    });
}

// --- Cancel ---
$('#cancel-btn').addEventListener('click', async () => {
    if (!currentJobId) return;
    try {
        await fetch(`/api/cancel/${currentJobId}`, { method: 'POST' });
    } catch (e) { /* ignore */ }
});

// --- SSE Progress (with polling fallback for Cloudflare tunnels) ---
let pollTimer = null;

function startSSE(jobId) {
    if (evtSource) evtSource.close();
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }

    let sseGotMessage = false;
    evtSource = new EventSource(`/api/status/${jobId}`);

    evtSource.onmessage = (e) => {
        sseGotMessage = true;
        handleStatusUpdate(jobId, JSON.parse(e.data));
    };

    evtSource.onerror = () => {
        evtSource.close();
        evtSource = null;
        if (!sseGotMessage) startPolling(jobId);
    };

    setTimeout(() => {
        if (!sseGotMessage && evtSource) {
            evtSource.close();
            evtSource = null;
            startPolling(jobId);
        }
    }, 3000);
}

function startPolling(jobId) {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/status-poll/${jobId}`);
            if (!resp.ok) return;
            handleStatusUpdate(jobId, await resp.json());
        } catch (e) { /* retry next tick */ }
    }, 1000);
}

function handleStatusUpdate(jobId, data) {
    updateProgress(data);

    if (data.stage === 'complete') {
        stopProgress();
        loadResult(jobId);
    } else if (data.stage === 'error') {
        stopProgress();
        showError(data.message);
        hide($('#progress-section'));
        $('#generate-btn').disabled = false;
    } else if (data.stage === 'cancelled') {
        stopProgress();
        hide($('#progress-section'));
        $('#generate-btn').disabled = false;
    }
}

function stopProgress() {
    if (evtSource) { evtSource.close(); evtSource = null; }
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

function updateProgress(data) {
    if (data.stage === 'queued') {
        show($('#queue-info'));
        const pos = data.queue_position;
        $('#queue-position-text').textContent = pos > 1
            ? `Pending (position ${pos}, ${pos - 1} job${pos > 2 ? 's' : ''} ahead)`
            : 'Pending (next)';
        $('#progress-bar').style.width = '0%';
        $('#progress-stage').textContent = '';
        $('#progress-percent').textContent = '';
    } else {
        hide($('#queue-info'));
        const pct = Math.round(data.progress * 100);
        $('#progress-bar').style.width = pct + '%';
        $('#progress-stage').textContent = data.message;
        $('#progress-percent').textContent = pct + '%';
    }
}

// --- Result ---
async function loadResult(jobId) {
    try {
        const resp = await fetch(`/api/result/${jobId}`);
        if (!resp.ok) throw new Error('Failed to load result');
        const data = await resp.json();

        displayResult(data);

        // Load HDR data for client-side tone mapping
        await loadHdrData(jobId);

        hide($('#progress-section'));
        $('#generate-btn').disabled = false;
    } catch (e) {
        showError('Failed to load result: ' + e.message);
        $('#generate-btn').disabled = false;
    }
}

function displayResult(data) {
    const a = data.analysis;
    $('#result-time').textContent = data.processing_time_seconds.toFixed(1) + 's';
    $('#result-dr').textContent = a.dynamic_range_ev.toFixed(1) + ' EV';

    // Percentiles
    const p = a.luminance_percentiles;
    if (p) {
        $('#result-percentiles').textContent =
            `${(p['50'] || 0).toFixed(4)} / ${(p['90'] || 0).toFixed(4)} / ${(p['99'] || 0).toFixed(4)}`;
    }

    // Fill comparison table (SDR input vs HDR output, both in linear domain)
    const inM = inputLinearMetrics;
    if (inM) {
        $('#cmp-input-mean').textContent = inM.mean.toFixed(4);
        $('#cmp-output-mean').textContent = a.mean_luminance.toFixed(4);
        const ratioMean = inM.mean > 0 ? a.mean_luminance / inM.mean : 0;
        $('#cmp-ratio-mean').textContent = ratioMean.toFixed(2) + 'x';

        $('#cmp-input-peak').textContent = inM.peak.toFixed(4);
        $('#cmp-output-peak').textContent = a.peak_luminance.toFixed(4);
        const ratioPeak = inM.peak > 0 ? a.peak_luminance / inM.peak : 0;
        $('#cmp-ratio-peak').textContent = ratioPeak.toFixed(2) + 'x';

        // Contrast ratio (P99.9 / P0.1)
        $('#cmp-input-contrast').textContent = formatContrast(inM.contrast);
        $('#cmp-output-contrast').textContent = formatContrast(a.contrast_ratio);
        const ratioContrast = inM.contrast > 0 ? a.contrast_ratio / inM.contrast : 0;
        $('#cmp-ratio-contrast').textContent = ratioContrast.toFixed(1) + 'x';
    } else {
        $('#cmp-input-mean').textContent = '—';
        $('#cmp-output-mean').textContent = a.mean_luminance.toFixed(4);
        $('#cmp-ratio-mean').textContent = '—';
        $('#cmp-input-peak').textContent = '—';
        $('#cmp-output-peak').textContent = a.peak_luminance.toFixed(4);
        $('#cmp-ratio-peak').textContent = '—';
        $('#cmp-input-contrast').textContent = '—';
        $('#cmp-output-contrast').textContent = formatContrast(a.contrast_ratio);
        $('#cmp-ratio-contrast').textContent = '—';
    }

    if (a.hdr_histogram && a.hdr_histogram.counts.length > 0) {
        drawHdrHistogram($('#hdr-histogram-canvas'), a.hdr_histogram);
    }

    $('#download-exr').href = data.download_url;

    if (inputLocalUrl) {
        $('#compare-input').src = inputLocalUrl;
    }

    show($('#result-section'));
    initCompareSlider();
}

function formatContrast(ratio) {
    if (ratio >= 1000) return (ratio / 1000).toFixed(1) + 'k:1';
    return ratio.toFixed(0) + ':1';
}

// --- Client-side Tone Mapping ---
async function loadHdrData(jobId) {
    try {
        const resp = await fetch(`/api/hdr-raw/${jobId}?max_dim=1024`);
        if (!resp.ok) return;

        const buffer = await resp.arrayBuffer();
        const view = new DataView(buffer);
        const width = view.getUint32(0, true);
        const height = view.getUint32(4, true);
        const pixels = new Float32Array(buffer, 8);

        hdrData = { width, height, pixels };
        applyClientTonemap();
    } catch (e) {
        console.warn('Failed to load HDR data for client-side tone mapping:', e);
    }
}

function applyClientTonemap() {
    if (!hdrData) return;

    const { width, height, pixels } = hdrData;
    const exposure = parseFloat($('#exposure-slider').value);
    const tonemap = $('#tonemap-select').value;
    const gamma = parseFloat($('#gamma-slider').value);
    const exposureMul = Math.pow(2, exposure);

    const canvas = $('#compare-canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);
    const out = imgData.data;

    for (let i = 0; i < width * height; i++) {
        let r = Math.max(0, pixels[i * 3]) * exposureMul;
        let g = Math.max(0, pixels[i * 3 + 1]) * exposureMul;
        let b = Math.max(0, pixels[i * 3 + 2]) * exposureMul;

        // Tone mapping
        if (tonemap === 'aces') {
            r = tonemapAces(r);
            g = tonemapAces(g);
            b = tonemapAces(b);
        } else if (tonemap === 'reinhard') {
            r = r / (1 + r);
            g = g / (1 + g);
            b = b / (1 + b);
        } else {
            r = Math.min(r, 1);
            g = Math.min(g, 1);
            b = Math.min(b, 1);
        }

        // Gamma
        r = applyGamma(r, gamma);
        g = applyGamma(g, gamma);
        b = applyGamma(b, gamma);

        const idx = i * 4;
        out[idx] = Math.round(Math.min(1, Math.max(0, r)) * 255);
        out[idx + 1] = Math.round(Math.min(1, Math.max(0, g)) * 255);
        out[idx + 2] = Math.round(Math.min(1, Math.max(0, b)) * 255);
        out[idx + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
}

function tonemapAces(x) {
    const a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return Math.min(1, Math.max(0, (x * (a * x + b)) / (x * (c * x + d) + e)));
}

function applyGamma(v, gamma) {
    if (Math.abs(gamma - 2.4) < 0.01) {
        // sRGB
        return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
    }
    return Math.pow(Math.max(0, v), 1 / gamma);
}

// --- A/B Comparison Slider ---
let compareInitialized = false;
function initCompareSlider() {
    const container = $('#compare-container');
    const slider = $('#compare-slider');
    const canvas = $('#compare-canvas');

    if (!compareInitialized) {
        let isDragging = false;

        function setPosition(x) {
            const rect = container.getBoundingClientRect();
            let pct = ((x - rect.left) / rect.width) * 100;
            pct = Math.max(0, Math.min(100, pct));
            slider.style.left = pct + '%';
            canvas.style.clipPath = `inset(0 0 0 ${pct}%)`;
        }

        container.addEventListener('mousedown', (e) => {
            isDragging = true;
            setPosition(e.clientX);
        });
        document.addEventListener('mousemove', (e) => {
            if (isDragging) setPosition(e.clientX);
        });
        document.addEventListener('mouseup', () => { isDragging = false; });

        container.addEventListener('touchstart', (e) => {
            isDragging = true;
            setPosition(e.touches[0].clientX);
        }, { passive: true });
        document.addEventListener('touchmove', (e) => {
            if (isDragging) setPosition(e.touches[0].clientX);
        }, { passive: true });
        document.addEventListener('touchend', () => { isDragging = false; });

        compareInitialized = true;
    }

    // Reset to 50%
    slider.style.left = '50%';
    canvas.style.clipPath = 'inset(0 0 0 50%)';
}

// --- Error Display ---
function showError(msg) {
    $('#error-text').textContent = msg;
    show($('#error-section'));
}

// --- Histogram Drawing ---
function drawHistogram(canvas, histData) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const channels = [
        { data: histData.r, color: 'rgba(255, 80, 80, 0.5)' },
        { data: histData.g, color: 'rgba(80, 220, 80, 0.5)' },
        { data: histData.b, color: 'rgba(80, 120, 255, 0.5)' },
    ];

    let maxVal = 0;
    for (const ch of channels) {
        for (let i = 1; i < 255; i++) {
            if (ch.data[i] > maxVal) maxVal = ch.data[i];
        }
    }

    if (maxVal === 0) return;

    const barW = w / 256;
    const plotH = h - 16;

    for (const ch of channels) {
        ctx.fillStyle = ch.color;
        ctx.beginPath();
        ctx.moveTo(0, plotH);
        for (let i = 0; i < 256; i++) {
            const barH = (ch.data[i] / maxVal) * plotH * 0.95;
            ctx.lineTo(i * barW, plotH - barH);
        }
        ctx.lineTo(w, plotH);
        ctx.closePath();
        ctx.fill();
    }

    ctx.fillStyle = 'rgba(139, 143, 163, 0.8)';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', 2, h - 2);
    ctx.textAlign = 'center';
    ctx.fillText('64', w * 0.25, h - 2);
    ctx.fillText('128', w * 0.5, h - 2);
    ctx.fillText('192', w * 0.75, h - 2);
    ctx.textAlign = 'right';
    ctx.fillText('255', w - 2, h - 2);
    ctx.textAlign = 'left';
}

function drawHdrHistogram(canvas, histData) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const counts = histData.counts;
    if (!counts.length) return;

    const maxCount = Math.max(...counts);
    if (maxCount === 0) return;

    const barW = w / counts.length;
    const plotH = h - 16;

    ctx.fillStyle = 'rgba(108, 138, 255, 0.6)';
    for (let i = 0; i < counts.length; i++) {
        const barH = (counts[i] / maxCount) * plotH * 0.95;
        ctx.fillRect(i * barW, plotH - barH, barW, barH);
    }

    const logMin = histData.log_min;
    const logMax = histData.log_max;
    const logRange = logMax - logMin;
    ctx.fillStyle = 'rgba(139, 143, 163, 0.8)';
    ctx.font = '9px monospace';

    const tickStart = Math.ceil(logMin);
    const tickEnd = Math.floor(logMax);
    for (let t = tickStart; t <= tickEnd; t++) {
        const xFrac = (t - logMin) / logRange;
        const x = xFrac * w;
        ctx.textAlign = 'center';
        ctx.fillText(t.toFixed(0), x, h - 2);
        ctx.fillRect(x, plotH, 1, 3);
    }

    ctx.textAlign = 'left';
    ctx.fillText(logMin.toFixed(1), 2, 10);
    ctx.textAlign = 'right';
    ctx.fillText(logMax.toFixed(1), w - 2, 10);
    ctx.textAlign = 'left';

    ctx.fillStyle = 'rgba(139, 143, 163, 0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('log\u2081\u2080(luminance)', w / 2, 10);
    ctx.textAlign = 'left';
}

// --- Utilities ---
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
