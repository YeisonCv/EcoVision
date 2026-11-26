const EcoVision=(()=>{
  const $=id=>document.getElementById(id);const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const MIB = 1024 * 1024;
  const INVALID_IMAGE = 'No se pudo leer la imagen. Usa un JPG o PNG válido.';
  const PROCESSING_ERROR = 'No se pudo procesar la imagen en este navegador.';

  function checkDimensions(width, height) {
    if (!width || !height || width * height > 40_000_000 || Math.max(width, height) > 16_384) {
      throw new Error('Las dimensiones de la imagen superan el límite seguro.');
    }
  }

  // Read dimensions before asking the browser to decode the image into memory.
  function readImageDimensions(data) {
    if (data.byteLength >= 24 && data.getUint32(0) === 0x89504e47 &&
        data.getUint32(4) === 0x0d0a1a0a && data.getUint32(12) === 0x49484452) {
      return [data.getUint32(16), data.getUint32(20)];
    }
    if (data.byteLength >= 2 && data.getUint16(0) === 0xffd8) {
      let offset = 2;
      while (offset + 4 <= data.byteLength && data.getUint8(offset++) === 0xff) {
        while (offset < data.byteLength && data.getUint8(offset) === 0xff) offset++;
        if (offset + 3 > data.byteLength) break;
        const marker = data.getUint8(offset++);
        if (marker === 0xda || marker === 0xd9) break;
        const length = data.getUint16(offset);
        if (length < 2 || offset + length > data.byteLength) break;
        if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker) && length >= 8) {
          return [data.getUint16(offset + 5), data.getUint16(offset + 3)];
        }
        offset += length;
      }
    }
    throw new Error(INVALID_IMAGE);
  }

  async function checkFile(source) {
    if (!source.size) throw new Error(INVALID_IMAGE);
    if (source.size > 25 * MIB) throw new Error('La imagen original supera el límite seguro de 25 MiB.');
    let dimensions;
    try {
      dimensions = readImageDimensions(new DataView(await source.arrayBuffer()));
    } catch {
      throw new Error(INVALID_IMAGE);
    }
    checkDimensions(...dimensions);
  }

  async function processImage(source, maxImageMiB) {
    let bitmap;
    let objectUrl;
    let canvas;
    try {
      let drawable = source;
      let width = source.videoWidth;
      let height = source.videoHeight;
      if (source instanceof Blob) {
        await checkFile(source);
        try {
          if (typeof createImageBitmap === 'function') {
            bitmap = await createImageBitmap(source);
            drawable = bitmap;
            width = bitmap.width;
            height = bitmap.height;
          } else {
            drawable = new Image();
            objectUrl = URL.createObjectURL(source);
            drawable.src = objectUrl;
            await drawable.decode();
            width = drawable.naturalWidth;
            height = drawable.naturalHeight;
          }
        } catch {
          throw new Error(INVALID_IMAGE);
        }
      } else if (!width || !height) {
        throw new Error('Espera a que la cámara esté lista para capturar.');
      }
      checkDimensions(width, height);
      canvas = document.createElement('canvas');
      const targetBytes = Math.floor(maxImageMiB * MIB * .95);
      let scale = Math.min(1, 1024 / Math.max(width, height));
      for (let pass = 0; pass < 8; pass++) {
        canvas.width = Math.max(1, Math.round(width * scale));
        canvas.height = Math.max(1, Math.round(height * scale));
        const ctx = canvas.getContext('2d', {alpha: false});
        if (!ctx) throw new Error(PROCESSING_ERROR);
        try {
          ctx.drawImage(drawable, 0, 0, canvas.width, canvas.height);
          for (const quality of [.85, .75, .65, .55, .45]) {
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
            if (!blob?.size || blob.type !== 'image/jpeg') throw new Error(PROCESSING_ERROR);
            if (blob.size <= targetBytes) return blob;
          }
        } catch {
          throw new Error(PROCESSING_ERROR);
        }
        scale *= .8;
      }
      throw new Error(`No se pudo comprimir la imagen para enviarla dentro del límite de ${maxImageMiB} MiB.`);
    } finally {
      if (bitmap) bitmap.close();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      if (canvas) canvas.width = canvas.height = 0;
    }
  }

  async function requestPrediction(image, filename, maxImageMiB) {
    const form = new FormData();
    form.append('image', image, filename);
    const response = await fetch('/api/predict', {method: 'POST', body: form});
    const data = await response.json().catch(() => null);
    if (response.status === 413) {
      throw new Error(data?.error || `La solicitud es demasiado grande. Envía una sola imagen de hasta ${maxImageMiB} MiB.`);
    }
    if (!response.ok || !data) throw new Error(data?.error || 'Ocurrió un error al analizar la imagen.');
    return data;
  }
  async function getStats(){const r=await fetch('/api/statistics');if(!r.ok)throw new Error('No se pudieron cargar los datos.');return r.json()}
  function drawBars(target,rows){const box=$(target);if(!box)return;box.innerHTML='';const max=Math.max(1,...rows.map(x=>x.count));rows.forEach(row=>{const line=document.createElement('div');line.className='bar-row';line.innerHTML=`<span>${escapeHtml(row.category)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(2,row.count/max*100)}%"></div></div><span class="bar-value">${row.count}</span>`;box.append(line)})}
  async function dashboard(){try{const d=await getStats();$('total-stat').textContent=d.total;$('common-stat').textContent=d.most_common||'Sin datos';$('latest-stat').textContent=d.latest?`${d.latest.category} · ${new Date(d.latest.created_at).toLocaleDateString('es-CO')}`:'Sin registros';drawBars('dashboard-chart',d.by_category);$('dashboard-empty').classList.toggle('hidden',d.total>0)}catch(e){$('dashboard-empty').textContent=e.message;$('dashboard-empty').classList.remove('hidden')}}
  function classifier() {
    const input = $('image-input');
    const preview = $('preview');
    const video = $('camera-video');
    const photo = $('camera-photo');
    const status = $('camera-status');
    const uploadButton = $('analyze-upload');
    const camButton = $('analyze-camera');
    const startButton = $('start-camera');
    const captureButton = $('capture-photo');
    const burstButton = $('burst-camera');
    const repeatButton = $('repeat-photo');
    const tabs = Array.from(document.querySelectorAll('.tab'));
    const maxImageMiB = Number(input.dataset.maxImageMib);
    let file, captured, stream, previewUrl, cameraPhotoUrl, resultImageUrl;
    let busy = false, preparing = false, starting = false, disposed = false, selection = 0, cameraAttempt = 0;

    function syncButtons() {
      input.disabled = busy || preparing;
      uploadButton.disabled = busy || !file;
      camButton.disabled = busy || !captured;
      captureButton.disabled = burstButton.disabled = busy || !stream;
      startButton.disabled = busy || starting || Boolean(stream);
      repeatButton.disabled = busy;
      $('new-analysis').disabled = busy;
      tabs.forEach(tab => tab.disabled = busy);
    }

    function clearPreview() {
      file = null;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
      preview.removeAttribute('src');
      $('preview-wrap').classList.add('hidden');
    }

    function clearFile() {
      selection++;
      input.value = '';
      clearPreview();
      syncButtons();
    }

    function clearCapture() {
      captured = null;
      if (cameraPhotoUrl) URL.revokeObjectURL(cameraPhotoUrl);
      cameraPhotoUrl = null;
      photo.removeAttribute('src');
      photo.classList.add('hidden');
      video.classList.remove('hidden');
      repeatButton.classList.add('hidden');
    }

    function stopCamera() {
      cameraAttempt++;
      starting = false;
      if (stream) stream.getTracks().forEach(track => track.stop());
      stream = null;
      video.srcObject = null;
      clearCapture();
      status.textContent = 'La cámara requiere permiso y conexión segura (HTTPS o localhost).';
      syncButtons();
    }

    function showError(message) {
      $('classify-error').textContent = message;
      $('classify-error').classList.remove('hidden');
    }

    async function run(action) {
      if (busy) return;
      busy = true;
      $('classify-error').classList.add('hidden');
      syncButtons();
      try {
        await action();
      } catch (error) {
        if (!disposed) showError(error.message);
      } finally {
        busy = false;
        syncButtons();
      }
    }

    input.addEventListener('change', async () => {
      const original = input.files[0];
      const id = ++selection;
      clearPreview();
      $('classify-error').classList.add('hidden');
      preparing = Boolean(original);
      syncButtons();
      if (!original) return;
      try {
        const prepared = await processImage(original, maxImageMiB);
        if (id !== selection || disposed) return;
        file = prepared;
        previewUrl = URL.createObjectURL(file);
        preview.src = previewUrl;
        $('preview-wrap').classList.remove('hidden');
      } catch (error) {
        if (id !== selection || disposed) return;
        clearFile();
        showError(error.message);
      } finally {
        preparing = false;
        syncButtons();
      }
    });
    $('remove-image').addEventListener('click', clearFile);

    tabs.forEach(tab => tab.addEventListener('click', () => {
      const cameraSelected = tab.dataset.tab === 'camera';
      tabs.forEach(item => item.classList.toggle('active', item === tab));
      $('upload-pane').classList.toggle('hidden', cameraSelected);
      $('camera-pane').classList.toggle('hidden', !cameraSelected);
      if (!cameraSelected) stopCamera();
    }));

    startButton.addEventListener('click', async () => {
      const attempt = ++cameraAttempt;
      starting = true;
      syncButtons();
      try {
        status.textContent = 'Solicitando permiso de cámara…';
        const opened = await navigator.mediaDevices.getUserMedia({video: {facingMode: {ideal: 'environment'}}, audio: false});
        if (attempt !== cameraAttempt) {
          opened.getTracks().forEach(track => track.stop());
          return;
        }
        stream = opened;
        video.srcObject = stream;
        status.textContent = 'Cámara activa. Puedes capturar una fotografía.';
      } catch {
        if (attempt === cameraAttempt) status.textContent = 'No se pudo acceder a la cámara. Puedes cargar una imagen manualmente.';
      } finally {
        if (attempt === cameraAttempt) {
          starting = false;
          syncButtons();
        }
      }
    });

    captureButton.addEventListener('click', () => run(async () => {
      const image = await processImage(video, maxImageMiB);
      if (disposed) return;
      clearCapture();
      captured = image;
      cameraPhotoUrl = URL.createObjectURL(captured);
      photo.src = cameraPhotoUrl;
      photo.classList.remove('hidden');
      video.classList.add('hidden');
      repeatButton.classList.remove('hidden');
      status.textContent = 'Captura lista para analizar.';
    }));

    repeatButton.addEventListener('click', () => {
      clearCapture();
      status.textContent = 'Cámara activa.';
      syncButtons();
    });

    burstButton.addEventListener('click', () => run(async () => {
      clearCapture();
      status.textContent = 'Capturando ráfaga de 3 fotografías…';
      const photos = [];
      let analyzed = 0;
      let failure;
      try {
        if (!video.videoWidth) throw new Error('Espera a que la cámara esté lista para capturar.');
        for (let i = 0; i < 3 && !disposed; i++) {
          photos.push(await processImage(video, maxImageMiB));
          if (i < 2) await new Promise(resolve => setTimeout(resolve, 500));
        }
      } catch (error) {
        failure = error;
      }
      for (const [i, image] of photos.entries()) {
        if (disposed) return;
        status.textContent = `Analizando fotografía ${i + 1} de ${photos.length}…`;
        try {
          await analyze(image, `captura-${i + 1}.jpg`);
          analyzed++;
        } catch (error) {
          failure = error;
        }
      }
      if (disposed) return;
      status.textContent = `Ráfaga terminada. Fotografías analizadas: ${analyzed} de 3.`;
      if (failure) throw failure;
    }));

    function clearResult() {
      if (resultImageUrl) URL.revokeObjectURL(resultImageUrl);
      resultImageUrl = null;
      $('result-image').removeAttribute('src');
      $('result-content').classList.add('hidden');
      $('result-placeholder').classList.remove('hidden');
    }

    async function analyze(image, filename) {
      const data = await requestPrediction(image, filename, maxImageMiB);
      if (disposed) return;
      clearResult();
      resultImageUrl = URL.createObjectURL(image);
      $('result-image').src = resultImageUrl;
      $('result-category').textContent = data.category;
      $('result-confidence').textContent = `${(data.confidence * 100).toFixed(1)}% de confianza`;
      $('result-note').textContent = data.message;
      $('result-top').innerHTML = data.top3.slice(1).map(item =>
        `<li>${escapeHtml(item.category)} · ${(item.confidence * 100).toFixed(1)}%</li>`).join('');
      $('result-placeholder').classList.add('hidden');
      $('result-content').classList.remove('hidden');
    }

    uploadButton.addEventListener('click', () => run(() => analyze(file, 'ecovision.jpg')));
    camButton.addEventListener('click', () => run(() => analyze(captured, 'captura.jpg')));
    $('new-analysis').addEventListener('click', () => {
      clearResult();
      clearFile();
      clearCapture();
      $('classify-error').classList.add('hidden');
      if (stream) status.textContent = 'Cámara activa.';
      syncButtons();
    });
    window.addEventListener('pagehide', () => {
      disposed = true;
      stopCamera();
      clearFile();
      clearResult();
    });
    window.addEventListener('pageshow', () => { disposed = false; });
  }
  let currentPage=1;async function history(){const fetchPage=async()=>{const params=new URLSearchParams({page:currentPage,per_page:20});if($('history-category').value)params.set('category',$('history-category').value);if($('date-from').value)params.set('date_from',$('date-from').value);if($('date-to').value)params.set('date_to',$('date-to').value);const r=await fetch(`/api/predictions?${params}`),d=await r.json();if(!r.ok)throw new Error(d.error);$('history-rows').innerHTML=d.items.map(x=>`<tr><td>${x.id}</td><td>${escapeHtml(x.filename)}</td><td>${escapeHtml(x.category)}</td><td>${(x.confidence*100).toFixed(1)}%</td><td>${new Date(x.created_at).toLocaleString('es-CO',{timeZone:'UTC'})}</td></tr>`).join('');$('history-empty').classList.toggle('hidden',d.items.length>0);$('page-label').textContent=`Página ${d.page} de ${Math.max(1,d.pages)}`;$('previous-page').disabled=d.page<=1;$('next-page').disabled=d.page>=d.pages};$('filter-history').addEventListener('click',()=>{currentPage=1;fetchPage()});$('previous-page').addEventListener('click',()=>{currentPage=Math.max(1,currentPage-1);fetchPage()});$('next-page').addEventListener('click',()=>{currentPage++;fetchPage()});try{await fetchPage()}catch(e){$('history-empty').textContent=e.message;$('history-empty').classList.remove('hidden')}}
  async function statistics(){try{const d=await getStats();$('total-stat').textContent=d.total;$('common-stat').textContent=d.most_common||'Sin datos';$('average-stat').textContent=d.average_confidence===null?'—':`${(d.average_confidence*100).toFixed(1)}%`;$('statistics-empty').classList.toggle('hidden',d.total>0);drawBars('statistics-chart',d.by_category);const max=Math.max(1,...d.daily.map(x=>x.count));$('daily-chart').innerHTML=d.daily.map(x=>`<div class="day-column"><div class="day-bar" style="height:${Math.max(2,x.count/max*90)}px"></div>${escapeHtml(x.date.slice(5))}<br>${x.count}</div>`).join('')}catch(e){$('statistics-empty').textContent=e.message;$('statistics-empty').classList.remove('hidden')}}
  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.querySelector('.menu-toggle');
    const nav = document.querySelector('.nav');
    toggle?.addEventListener('click', () => toggle.setAttribute('aria-expanded', nav.classList.toggle('open')));
    nav?.querySelectorAll('a').forEach(link => link.classList.toggle('active', link.dataset.route === location.pathname));
    const initialize = new Map([['/', dashboard], ['/clasificador', classifier], ['/historial', history], ['/estadisticas', statistics]]);
    initialize.get(location.pathname)?.();
  });
  return {processImage, requestPrediction};
})();
