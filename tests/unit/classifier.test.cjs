const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.join(__dirname, '../../visualizacion/static/js/app.js'), 'utf8');

function element() {
  const events = new Map();
  return {
    dataset: {}, classList: {add() {}, remove() {}, toggle() {}},
    addEventListener: (name, callback) => events.set(name, callback),
    trigger: (name) => events.get(name)(),
    removeAttribute() {},
  };
}

function browser({encodedSize = () => 100, fetch, bitmap, pathname = '/clasificador'} = {}) {
  const elements = new Map();
  const calls = [];
  const canvases = [];
  const tab = element();
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  get('image-input').dataset.maxImageMib = '4';
  Object.assign(get('camera-video'), {videoWidth: 1920, videoHeight: 1080});
  let stopCount = 0;
  const stop = () => stopCount++;
  const trackStream = {getTracks: () => [{stop}]};
  const events = new Map();
  const context = {
    Blob, FormData, URL, fetch, location: {pathname},
    navigator: {mediaDevices: {getUserMedia: async () => trackStream}},
    createImageBitmap: bitmap,
    setTimeout: callback => callback(),
    window: {addEventListener: (name, callback) => events.set(name, callback)},
    document: {
      getElementById: get,
      querySelector: () => null,
      querySelectorAll: () => [tab],
      addEventListener: (name, callback) => events.set(name, callback),
      createElement: () => {
        const canvas = {
          getContext: () => ({drawImage() {}}),
          toBlob(callback, type, quality) {
            const call = {width: this.width, height: this.height, type, quality};
            calls.push(call);
            const size = encodedSize(call);
            callback(size ? new Blob([new Uint8Array(size)], {type}) : null);
          },
        };
        canvases.push(canvas);
        return canvas;
      },
    },
  };
  const api = vm.runInNewContext(source + '\nEcoVision', context);
  return {api, get, calls, canvases, stopCount: () => stopCount, trigger: name => events.get(name)()};
}

function pngSource(width, height, size = 24) {
  const bytes = Buffer.alloc(size);
  Buffer.from('89504e470d0a1a0a0000000d49484452', 'hex').copy(bytes);
  bytes.writeUInt32BE(width, 16);
  bytes.writeUInt32BE(height, 20);
  return new Blob([bytes], {type: 'image/png'});
}

test('accepts a source over 4 MiB, resizes proportionally and releases decode resources', async () => {
  let closed = false;
  const page = browser({bitmap: async () => ({width: 1920, height: 1080, close: () => { closed = true; }})});
  const image = await page.api.processImage(pngSource(1920, 1080, 5 * 1024 * 1024), 4);
  assert.equal(image.type, 'image/jpeg');
  assert.deepEqual(page.calls, [{width: 1024, height: 576, type: 'image/jpeg', quality: .85}]);
  assert.ok(closed);
  assert.equal(page.canvases[0].width, 0);
  assert.equal(page.canvases[0].height, 0);
});

test('reduces quality and then dimensions if encoded blobs exceed the target', async () => {
  const page = browser({encodedSize: ({width}) => width < 1024 ? 200 : 1024 * 1024});
  const image = await page.api.processImage(page.get('camera-video'), 1);
  assert.equal(image.size, 200);
  assert.deepEqual(page.calls.map(({width, height, quality}) => [width, height, quality]), [
    [1024, 576, .85], [1024, 576, .75], [1024, 576, .65],
    [1024, 576, .55], [1024, 576, .45], [819, 461, .85],
  ]);
});

test('keeps small images small and releases canvas on encoder error', async () => {
  const page = browser({encodedSize: () => 0});
  await assert.rejects(page.api.processImage({videoWidth: 240, videoHeight: 320}, 4), /No se pudo procesar/);
  assert.equal(page.calls[0].width, 240);
  assert.equal(page.calls[0].height, 320);
  assert.equal(page.canvases[0].width, 0);
});

test('rejects invalid or excessive originals before decoding', async () => {
  const page = browser({bitmap: async () => assert.fail('image must be rejected before decode')});
  await assert.rejects(page.api.processImage(new Blob(['not an image']), 4), /JPG o PNG válido/);
  await assert.rejects(page.api.processImage(pngSource(100_000, 100_000), 4), /dimensiones/);
  await assert.rejects(page.api.processImage(pngSource(400, 300, 25 * 1024 * 1024 + 1), 4), /25 MiB/);
});

test('reports JSON and non-JSON 413 responses with actionable messages', async () => {
  const body = new Blob(['jpeg'], {type: 'image/jpeg'});
  const request = response => browser({fetch: async () => response}).api.requestPrediction(body, 'capture.jpg', 4);
  await assert.rejects(request(new Response('FUNCTION_PAYLOAD_TOO_LARGE', {status: 413})), /4 MiB/);
  await assert.rejects(request(Response.json({error: 'El archivo supera el límite.'}, {status: 413})), /El archivo supera/);
});

test('upload, capture and burst each send one processed JPEG per prediction', async () => {
  const uploaded = [];
  const page = browser({
    bitmap: async () => ({width: 40, height: 30, close() {}}),
    fetch: async (url, options) => {
      assert.equal(url, '/api/predict');
      assert.equal(options.method, 'POST');
      assert.deepEqual([...options.body.keys()], ['image']);
      const image = options.body.get('image');
      uploaded.push([image.name, image.type, image.size]);
      return Response.json({category: 'paper', confidence: .8, top3: [], message: ''}, {status: 201});
    },
  });
  page.trigger('DOMContentLoaded');
  page.get('image-input').files = [pngSource(40, 30)];
  await page.get('image-input').trigger('change');
  await page.get('analyze-upload').trigger('click');
  await page.get('start-camera').trigger('click');
  await page.get('capture-photo').trigger('click');
  await page.get('analyze-camera').trigger('click');
  await page.get('burst-camera').trigger('click');
  assert.deepEqual(uploaded, ['ecovision.jpg', 'captura.jpg', 'captura-1.jpg', 'captura-2.jpg', 'captura-3.jpg']
    .map(filename => [filename, 'image/jpeg', 100]));
  assert.match(page.get('camera-status').textContent, /3 de 3/);
  assert.equal(page.calls.length, 5);
  page.trigger('pagehide');
  assert.equal(page.stopCount(), 1);
});
