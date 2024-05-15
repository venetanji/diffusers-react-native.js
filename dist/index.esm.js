// src/hub/browser.ts
import { downloadFile } from "@huggingface/hub";

// src/hub/indexed-db.ts
import { openDB } from "idb";

// src/util/Tensor.ts
import { Tensor } from "@xenova/transformers";
import seedrandom from "seedrandom";
Tensor.prototype.reverse = function() {
  return new Tensor(this.type, this.data.reverse(), this.dims.slice());
};
Tensor.prototype.sub = function(value) {
  return this.clone().sub_(value);
};
Tensor.prototype.sub_ = function(value) {
  if (typeof value === "number") {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error("Cannot subtract tensors of different sizes");
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] -= value.data[i];
    }
  } else {
    throw new Error("Invalid argument");
  }
  return this;
};
Tensor.prototype.add = function(value) {
  return this.clone().add_(value);
};
Tensor.prototype.add_ = function(value) {
  if (typeof value === "number") {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error("Cannot subtract tensors of different sizes");
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += value.data[i];
    }
  } else {
    throw new Error("Invalid argument");
  }
  return this;
};
Tensor.prototype.cumprod = function(dim) {
  return this.clone().cumprod_(dim);
};
Tensor.prototype.cumprod_ = function(dim) {
  const newDims = this.dims.slice();
  if (dim === void 0) {
    dim = this.dims.length - 1;
  }
  if (dim < 0 || dim >= this.dims.length) {
    throw new Error(`Invalid dimension: ${dim}`);
  }
  const size = newDims[dim];
  for (let i = 1; i < size; ++i) {
    for (let j = 0; j < this.data.length / size; ++j) {
      const index = j * size + i;
      this.data[index] *= this.data[index - 1];
    }
  }
  return this;
};
Tensor.prototype.mul = function(value) {
  return this.clone().mul_(value);
};
Tensor.prototype.mul_ = function(value) {
  if (typeof value === "number") {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error("Cannot multiply tensors of different sizes");
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] *= value.data[i];
    }
  } else {
    throw new Error("Invalid argument");
  }
  return this;
};
Tensor.prototype.div = function(value) {
  return this.clone().div_(value);
};
Tensor.prototype.div_ = function(value) {
  if (typeof value === "number") {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value;
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error("Cannot multiply tensors of different sizes");
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] /= value.data[i];
    }
  } else {
    throw new Error("Invalid argument");
  }
  return this;
};
Tensor.prototype.pow = function(value) {
  return this.clone().pow_(value);
};
Tensor.prototype.pow_ = function(value) {
  if (typeof value === "number") {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value);
    }
  } else if (value instanceof Tensor) {
    if (!this.dims === value.dims) {
      throw new Error("Cannot multiply tensors of different sizes");
    }
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = Math.pow(this.data[i], value.data[i]);
    }
  } else {
    throw new Error("Invalid argument");
  }
  return this;
};
Tensor.prototype.round = function() {
  return this.clone().round_();
};
Tensor.prototype.round_ = function() {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.round(this.data[i]);
  }
  return this;
};
Tensor.prototype.tile = function(reps) {
  return this.clone().tile_(reps);
};
Tensor.prototype.tile_ = function(reps) {
  if (typeof reps === "number") {
    reps = [reps];
  }
  if (reps.length < this.dims.length) {
    throw new Error("Invalid number of repetitions");
  }
  const newDims = [];
  const newStrides = [];
  for (let i = 0; i < this.dims.length; ++i) {
    newDims.push(this.dims[i] * reps[i]);
    newStrides.push(this.strides[i]);
  }
  const newData = new this.data.constructor(newDims.reduce((a, b) => a * b));
  for (let i = 0; i < newData.length; ++i) {
    let index = 0;
    for (let j = 0; j < this.dims.length; ++j) {
      index += Math.floor(i / newDims[j]) * this.strides[j];
    }
    newData[i] = this.data[index];
  }
  return new Tensor(this.type, newData, newDims);
};
Tensor.prototype.clipByValue = function(min, max) {
  return this.clone().clipByValue_(min, max);
};
Tensor.prototype.clipByValue_ = function(min, max) {
  if (max < min) {
    throw new Error("Invalid arguments");
  }
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.min(Math.max(this.data[i], min), max);
  }
  return this;
};
Tensor.prototype.exp = function() {
  return this.clone().exp_();
};
Tensor.prototype.exp_ = function() {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.exp(this.data[i]);
  }
  return this;
};
Tensor.prototype.sin = function() {
  return this.clone().sin_();
};
Tensor.prototype.sin_ = function() {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.sin(this.data[i]);
  }
  return this;
};
Tensor.prototype.cos = function() {
  return this.clone().cos_();
};
Tensor.prototype.cos_ = function() {
  for (let i = 0; i < this.data.length; ++i) {
    this.data[i] = Math.cos(this.data[i]);
  }
  return this;
};
Tensor.prototype.location = "cpu";
function range(start, end, step = 1, type = "float32") {
  const data = [];
  for (let i = start; i < end; i += step) {
    data.push(i);
  }
  return new Tensor(type, data, [data.length]);
}
function linspace(start, end, num, type = "float32") {
  const arr = [];
  const step = (end - start) / (num - 1);
  for (let i = 0; i < num; i++) {
    arr.push(start + step * i);
  }
  return new Tensor(type, arr, [num]);
}
function randomNormal(rng) {
  let u = 0;
  let v = 0;
  while (u === 0)
    u = rng();
  while (v === 0)
    v = rng();
  const num = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  return num;
}
function randomNormalTensor(shape, mean = 0, std = 1, type = "float32", seed = "") {
  const data = [];
  const rng = seed !== "" ? seedrandom(seed) : seedrandom();
  for (let i = 0; i < shape.reduce((a, b) => a * b); i++) {
    data.push(randomNormal(rng) * std + mean);
  }
  return new Tensor(type, data, shape);
}
function cat(tensors, axis = 0) {
  if (tensors.length === 0) {
    throw new Error("No tensors provided.");
  }
  if (axis < 0) {
    axis = tensors[0].dims.length + axis;
  }
  const tensorType = tensors[0].type;
  const tensorShape = [...tensors[0].dims];
  for (const t of tensors) {
    for (let i = 0; i < tensorShape.length; i++) {
      if (i !== axis && tensorShape[i] !== t.dims[i]) {
        throw new Error("Tensor dimensions must match for concatenation, except along the specified axis.");
      }
    }
  }
  tensorShape[axis] = tensors.reduce((sum, t) => sum + t.dims[axis], 0);
  const total = tensorShape.reduce((product, size) => product * size, 1);
  const data = new tensors[0].data.constructor(total);
  let offset = 0;
  for (const t of tensors) {
    const copySize = t.data.length / t.dims[axis];
    for (let i = 0; i < t.dims[axis]; i++) {
      const sourceStart = i * copySize;
      const sourceEnd = sourceStart + copySize;
      data.set(t.data.slice(sourceStart, sourceEnd), offset);
      offset += copySize;
    }
  }
  return new Tensor(tensorType, data, tensorShape);
}
function replaceTensors(modelRunResult) {
  const result = {};
  for (const prop in modelRunResult) {
    if (modelRunResult[prop].dims) {
      result[prop] = new Tensor(
        // @ts-ignore
        modelRunResult[prop].type,
        // @ts-ignore
        modelRunResult[prop].data,
        // @ts-ignore
        modelRunResult[prop].dims
      );
    }
  }
  return result;
}

// src/hub/index.ts
var cacheImpl = null;
function setCacheImpl(impl) {
  cacheImpl = impl;
}
async function getModelFile(modelRepoOrPath, fileName, fatal = true, options = {}) {
  return cacheImpl.getModelFile(modelRepoOrPath, fileName, fatal, options);
}
function getModelTextFile(modelPath, fileName, fatal, options) {
  return getModelFile(modelPath, fileName, fatal, { ...options, returnText: true });
}
async function getModelJSON(modelPath, fileName, fatal = true, options = {}) {
  const jsonData = await getModelTextFile(modelPath, fileName, fatal, options);
  return JSON.parse(jsonData);
}

// src/backends/index.ts
import * as ORT from "onnxruntime-react-native";
var ONNX = ORT.default ?? ORT;
var isNode = typeof process !== "undefined" && process?.release?.name === "node";
var Session = class _Session {
  session;
  config;
  constructor(session, config = {}, gpuEnable = false) {
    this.session = session;
    this.config = config || {};
  }
  static async create(modelOrPath, config, gpuEnable, options) {
    const arg = typeof modelOrPath === "string" ? modelOrPath : new Uint8Array(modelOrPath);
    console.log("Creating onnx session");
    const session = ONNX.InferenceSession.create(arg);
    return new _Session(await session, config);
  }
  async run(inputs) {
    const result = await this.session.run(inputs);
    return replaceTensors(result);
  }
  release() {
    return this.session.release();
  }
};

// src/pipelines/common.ts
async function sessionRun(session, inputs) {
  const result = await session.run(inputs);
  return replaceTensors(result);
}
var ProgressStatus = /* @__PURE__ */ ((ProgressStatus2) => {
  ProgressStatus2["Downloading"] = "Downloading";
  ProgressStatus2["Ready"] = "Ready";
  ProgressStatus2["Error"] = "Error";
  ProgressStatus2["EncodingImg2Img"] = "EncodingImg2Img";
  ProgressStatus2["EncodingPrompt"] = "EncodingPrompt";
  ProgressStatus2["RunningUnet"] = "RunningUnet";
  ProgressStatus2["RunningVae"] = "RunningVae";
  ProgressStatus2["Done"] = "Done";
  return ProgressStatus2;
})(ProgressStatus || {});
function setStatusText(payload) {
  switch (payload.status) {
    case "Downloading" /* Downloading */:
      return `Downloading ${payload.downloadStatus.file} (${Math.round(payload.downloadStatus.downloaded / payload.downloadStatus.size * 100)}%)`;
    case "EncodingImg2Img" /* EncodingImg2Img */:
      return `Encoding input image`;
    case "EncodingPrompt" /* EncodingPrompt */:
      return `Encoding prompt`;
    case "RunningUnet" /* RunningUnet */:
      return `Running UNet (${payload.unetTimestep}/${payload.unetTotalSteps})`;
    case "RunningVae" /* RunningVae */:
      return `Running VAE`;
    case "Done" /* Done */:
      return `Done`;
    case "Ready" /* Ready */:
      return `Ready`;
  }
  return "";
}
function dispatchProgress(cb, payload) {
  if (!payload.statusText) {
    payload.statusText = setStatusText(payload);
  }
  if (cb) {
    return cb(payload);
  }
}
async function loadModel(modelRepoOrPath, filename, opts, gpuEnable) {
  console.log("loading model");
  const model = await getModelFile(modelRepoOrPath, filename, true, opts);
  const dirName = filename.split("/")[0];
  const config = await getModelJSON(modelRepoOrPath, dirName + "/config.json", false, opts);
  return Session.create(model, config);
}

// src/hub/indexed-db.ts
var DEFAULT_CHUNK_LENGTH = 1024 * 1024 * 512;
var DbCache = class {
  dbName = "diffusers-cache";
  dbVersion = 1;
  db;
  init = async () => {
    const openRequest = await openDB(this.dbName, this.dbVersion, {
      upgrade(db) {
        if (!db.objectStoreNames.contains("files")) {
          db.createObjectStore("files");
        }
      }
    });
    this.db = openRequest;
  };
  storeFile = async (file, name, chunkLength = DEFAULT_CHUNK_LENGTH) => {
    const transaction = this.db.transaction(["files"], "readwrite");
    const store = transaction.objectStore("files");
    const chunks = Math.ceil(file.byteLength / chunkLength);
    const fileMetadata = {
      chunks,
      chunkLength,
      totalLength: file.byteLength
    };
    for (let i = 0; i < chunks; i++) {
      const chunk = file.slice(i * chunkLength, (i + 1) * chunkLength);
      const nameSuffix = i > 0 ? `-${i}` : "";
      const thisChunkLength = chunk.byteLength;
      await store.put({ ...fileMetadata, chunkLength: thisChunkLength, file: chunk, chunk: i }, `${name}${nameSuffix}`);
    }
    await transaction.done;
  };
  retrieveFile = async (filename, progressCallback, displayName) => {
    const transaction = this.db.transaction(["files"], "readonly");
    const store = transaction.objectStore("files");
    const request = await store.get(filename);
    if (!request) {
      return null;
    }
    if (request.chunks === 1) {
      return request;
    }
    let buffer;
    if (request.totalLength > 2 * 1024 * 1024 * 1024) {
      const memory = new WebAssembly.Memory({ initial: Math.ceil(request.totalLength / 65536), index: "i64" });
      buffer = memory.buffer;
    } else {
      buffer = new ArrayBuffer(request.totalLength);
    }
    const baseChunkLength = request.chunkLength;
    let view = new Uint8Array(buffer, 0, request.chunkLength);
    view.set(new Uint8Array(request.file));
    await dispatchProgress(progressCallback, {
      status: "Downloading" /* Downloading */,
      downloadStatus: {
        file: displayName,
        size: request.totalLength,
        downloaded: request.chunkLength
      }
    });
    for (let i = 1; i < request.chunks; i++) {
      const file = await store.get(`${filename}-${i}`);
      view = new Uint8Array(buffer, i * baseChunkLength, file.file.byteLength);
      view.set(new Uint8Array(file.file));
      await dispatchProgress(progressCallback, {
        status: "Downloading" /* Downloading */,
        downloadStatus: {
          file: displayName,
          size: request.totalLength,
          downloaded: i * baseChunkLength + file.file.byteLength
        }
      });
    }
    await transaction.done;
    return { ...request, file: buffer };
  };
};

// src/hub/common.ts
function pathJoin(...parts) {
  parts = parts.map((part, index) => {
    if (index) {
      part = part.replace(/^\//, "");
    }
    if (index !== parts.length - 1) {
      part = part.replace(/\/$/, "");
    }
    return part;
  });
  return parts.filter((p) => p !== "").join("/");
}

// src/hub/browser.ts
var cacheDir = "";
function setModelCacheDir(dir) {
  cacheDir = dir;
}
function getCacheKey(modelRepoOrPath, fileName, revision) {
  return pathJoin(cacheDir, modelRepoOrPath, revision === "main" ? "" : revision, fileName);
}
async function getModelFile2(modelRepoOrPath, fileName, fatal = true, options = {}) {
  const revision = options.revision || "main";
  const cachePath = getCacheKey(modelRepoOrPath, fileName, revision);
  const cache = new DbCache();
  await cache.init();
  const cachedData = await cache.retrieveFile(cachePath, options.progressCallback, fileName);
  if (cachedData) {
    if (options.returnText) {
      const decoder = new TextDecoder("utf-8");
      return decoder.decode(cachedData.file);
    }
    return cachedData.file;
  }
  let response;
  if (cacheDir) {
    response = await fetch(cachePath);
    if (!response || !response.body || response.status !== 200 || response.headers.get("content-type")?.startsWith("text/html")) {
      response = null;
    }
  }
  try {
    if (!response) {
      response = await downloadFile({ repo: modelRepoOrPath, path: fileName, revision });
    }
    if (!response || !response.body || response.status !== 200 || response.headers.get("content-type")?.startsWith("text/html")) {
      throw new Error(`Error downloading ${fileName}`);
    }
    const buffer = await readResponseToBuffer(response, options.progressCallback, fileName);
    await cache.storeFile(buffer, cachePath);
    if (options.returnText) {
      const decoder = new TextDecoder("utf-8");
      return decoder.decode(buffer);
    }
    return buffer;
  } catch (e) {
    if (!fatal) {
      return null;
    }
    throw e;
  }
}
function readResponseToBuffer(response, progressCallback, displayName) {
  const contentLength = response.headers.get("content-length");
  if (!contentLength) {
    return response.arrayBuffer();
  }
  let buffer;
  const contentLengthNum = parseInt(contentLength, 10);
  if (contentLengthNum > 2 * 1024 * 1024 * 1024) {
    const memory = new WebAssembly.Memory({ initial: Math.ceil(contentLengthNum / 65536), index: "i64" });
    buffer = memory.buffer;
  } else {
    buffer = new ArrayBuffer(contentLengthNum);
  }
  let offset = 0;
  return new Promise((resolve, reject) => {
    const reader = response.body.getReader();
    async function pump() {
      const { done, value } = await reader.read();
      if (done) {
        return resolve(buffer);
      }
      const chunk = new Uint8Array(buffer, offset, value.byteLength);
      chunk.set(new Uint8Array(value));
      offset += value.byteLength;
      await dispatchProgress(progressCallback, {
        status: "Downloading" /* Downloading */,
        downloadStatus: {
          file: displayName,
          size: contentLengthNum,
          downloaded: offset
        }
      });
      return pump();
    }
    pump().catch(reject);
  });
}
var browser_default = {
  getModelFile: getModelFile2
};

// src/schedulers/common.ts
import { Tensor as Tensor2 } from "@xenova/transformers";
function betasForAlphaBar(numDiffusionTimesteps, maxBeta = 0.999, alphaTransformType = "cosine") {
  function alphaBar(timeStep) {
    if (alphaTransformType === "cosine") {
      return Math.cos((timeStep + 8e-3) / 1.008 * Math.PI / 2) ** 2;
    } else if (alphaTransformType === "exp") {
      return Math.exp(timeStep * -12);
    }
    throw new Error("Unsupported alphaTransformType: " + alphaTransformType);
  }
  const betas = [];
  for (let i = 0; i < numDiffusionTimesteps; i++) {
    const t1 = i / numDiffusionTimesteps;
    const t2 = (i + 1) / numDiffusionTimesteps;
    betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta));
  }
  return new Tensor2(betas);
}

// src/schedulers/SchedulerBase.ts
var SchedulerBase = class {
  betas;
  alphas;
  alphasCumprod;
  finalAlphaCumprod;
  config;
  timesteps;
  numInferenceSteps = 20;
  constructor(config) {
    if (config.trained_betas !== null) {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps);
    } else if (config.beta_schedule === "linear") {
      this.betas = linspace(config.beta_start, config.beta_end, config.num_train_timesteps);
    } else if (config.beta_schedule === "scaled_linear") {
      this.betas = linspace(config.beta_start ** 0.5, config.beta_end ** 0.5, config.num_train_timesteps).pow(2);
    } else if (config.beta_schedule === "squaredcos_cap_v2") {
      this.betas = betasForAlphaBar(config.num_train_timesteps);
    } else {
      throw new Error(`${config.beta_schedule} does is not implemented for ${this.constructor}`);
    }
    this.timesteps = range(0, config.num_train_timesteps).reverse();
    this.alphas = linspace(1, 1, config.num_train_timesteps).sub(this.betas);
    this.alphasCumprod = this.alphas.cumprod();
    this.finalAlphaCumprod = config.set_alpha_to_one ? 1 : this.alphasCumprod[0].data;
    this.config = config;
  }
  scaleModelInput(input, timestep) {
    return input;
  }
  addNoise(originalSamples, noise, timestep) {
    const sqrtAlphaProd = this.alphasCumprod.data[timestep] ** 0.5;
    const sqrtOneMinusAlphaProd = (1 - this.alphasCumprod.data[timestep]) ** 0.5;
    return originalSamples.mul(sqrtAlphaProd).add(noise.mul(sqrtOneMinusAlphaProd));
  }
};

// src/schedulers/LCMScheduler.ts
import { Tensor as Tensor3 } from "@xenova/transformers";
var LCMScheduler = class extends SchedulerBase {
  initNoiseSigma;
  timeIndex;
  constructor(config) {
    super({
      rescale_betas_zero_snr: false,
      beta_start: 1e-4,
      beta_end: 0.02,
      beta_schedule: "linear",
      clip_sample: true,
      set_alpha_to_one: true,
      steps_offset: 0,
      prediction_type: "epsilon",
      thresholding: false,
      original_inference_steps: 50,
      ...config
    });
    this.initNoiseSigma = 1;
  }
  getVariance(timestep, prevTimestep) {
    const alphaProdT = this.alphasCumprod.data[timestep];
    const alphaProdTPrev = prevTimestep >= 0 ? this.alphasCumprod.data[prevTimestep] : this.finalAlphaCumprod;
    const betaProdT = 1 - alphaProdT;
    const betaProdTPrev = 1 - alphaProdTPrev;
    return betaProdTPrev / betaProdT * (1 - alphaProdT / alphaProdTPrev);
  }
  getScalingsForBoundaryConditionDiscrete(timestep) {
    const sigmaData = 0.5;
    const cSkip = sigmaData ** 2 / ((timestep / 0.1) ** 2 + sigmaData ** 2);
    const cOut = timestep / 0.1 / ((timestep / 0.1) ** 2 + sigmaData ** 2) ** 0.5;
    return [cSkip, cOut];
  }
  step(modelOutput, timestep, sample) {
    if (!this.numInferenceSteps) {
      throw new Error("numInferenceSteps is not set");
    }
    const prevTimeIndex = this.timeIndex + 1;
    let prevTimeStep;
    if (prevTimeIndex < this.timesteps.data.length) {
      prevTimeStep = this.timesteps.data[prevTimeIndex];
    } else {
      prevTimeStep = timestep;
    }
    const alphaProdT = this.alphasCumprod[timestep].data[0];
    const alphaProdTPrev = prevTimeStep >= 0 ? this.alphasCumprod[prevTimeStep].data[0] : this.finalAlphaCumprod;
    const betaProdT = 1 - alphaProdT;
    const betaProdTPrev = 1 - alphaProdTPrev;
    const [cSkip, cOut] = this.getScalingsForBoundaryConditionDiscrete(timestep);
    let predX0;
    const parametrization = this.config.prediction_type;
    if (parametrization === "epsilon") {
      predX0 = sample.sub(
        modelOutput.mul(Math.sqrt(betaProdT))
      ).div(Math.sqrt(alphaProdT));
    } else if (parametrization === "sample") {
      predX0 = sample;
    } else if (parametrization === "v_prediction") {
      predX0 = sample.mul(Math.sqrt(alphaProdT)).sub(modelOutput.mul(Math.sqrt(betaProdT)));
    }
    const denoised = predX0.mul(cOut).add(sample.mul(cSkip));
    let prevSample = denoised;
    if (this.timesteps.data.length > 1) {
      const noise = randomNormalTensor(modelOutput.dims);
      prevSample = denoised.mul(Math.sqrt(alphaProdTPrev)).add(noise.mul(Math.sqrt(betaProdTPrev)));
    }
    this.timeIndex++;
    return [
      prevSample,
      denoised
    ];
  }
  setTimesteps(numInferenceSteps) {
    this.numInferenceSteps = numInferenceSteps;
    if (this.numInferenceSteps > this.config.num_train_timesteps) {
      throw new Error("numInferenceSteps must be less than or equal to num_train_timesteps");
    }
    const lcmOriginSteps = this.config.original_inference_steps;
    const c = this.config.num_train_timesteps / lcmOriginSteps;
    const lcmOriginTimesteps = [];
    for (let i = 1; i <= lcmOriginSteps; i++) {
      lcmOriginTimesteps.push(i * c - 1);
    }
    const skippingStep = Math.floor(lcmOriginTimesteps.length / numInferenceSteps);
    const timesteps = [];
    for (let i = lcmOriginTimesteps.length - 1; i >= 0; i -= skippingStep) {
      timesteps.push(lcmOriginTimesteps[i]);
    }
    this.timeIndex = 0;
    this.timesteps = new Tensor3(
      new Int32Array(
        timesteps
      )
    );
  }
};

// src/tokenizers/CLIPTokenizer.ts
import { PreTrainedTokenizer, Tensor as Tensor4 } from "@xenova/transformers";
var CLIPTokenizer = class _CLIPTokenizer extends PreTrainedTokenizer {
  bos_token_id;
  eos_token_id;
  constructor(tokenizerJSON, tokenizerConfig) {
    super(tokenizerJSON, tokenizerConfig);
    this.added_tokens_regex = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/gui;
    const bos_token = this.getToken(tokenizerConfig, "bos_token");
    if (bos_token) {
      this.bos_token_id = this.model.tokens_to_ids.get(bos_token);
    }
    const eos_token = this.getToken(tokenizerConfig, "eos_token");
    if (eos_token) {
      this.eos_token_id = this.model.tokens_to_ids.get(eos_token);
    }
  }
  _call(text, {
    text_pair = null,
    // add_special_tokens = true, // TODO
    padding = false,
    truncation = false,
    max_length = null,
    return_tensor = true,
    // Different to HF
    return_tensor_dtype = "int64"
  } = {}) {
    let tokens;
    if (Array.isArray(text)) {
      if (text.length === 0) {
        throw Error("text array must be non-empty");
      }
      if (text_pair !== null) {
        if (!Array.isArray(text_pair)) {
          throw Error("text_pair must also be an array");
        } else if (text.length !== text_pair.length) {
          throw Error("text and text_pair must have the same length");
        }
        tokens = text.map(
          (t, i) => this.encode(t, text_pair[i])
        );
      } else {
        tokens = text.map((x) => this.encode(x));
      }
    } else {
      if (text === null) {
        throw Error("text may not be null");
      }
      if (Array.isArray(text_pair)) {
        throw Error("When specifying `text_pair`, since `text` is a string, `text_pair` must also be a string (i.e., not an array).");
      }
      tokens = [this.encode(text, text_pair)];
    }
    const maxLengthOfBatch = Math.max(...tokens.map((x) => x.length));
    if (max_length === null) {
      max_length = maxLengthOfBatch;
    }
    max_length = Math.min(max_length, this.model_max_length);
    if (this.bos_token_id) {
      tokens = tokens.map((x) => [this.bos_token_id].concat(x));
    }
    if (this.eos_token_id) {
      tokens = tokens.map((x) => x.concat([this.eos_token_id]));
    }
    let attention_mask = [];
    if (padding || truncation) {
      for (let i = 0; i < tokens.length; ++i) {
        if (tokens[i].length === max_length) {
          attention_mask.push(new Array(tokens[i].length).fill(1));
          continue;
        } else if (tokens[i].length > max_length) {
          if (truncation) {
            tokens[i] = tokens[i].slice(0, max_length);
          }
          attention_mask.push(new Array(tokens[i].length).fill(1));
        } else {
          if (padding) {
            const diff = max_length - tokens[i].length;
            if (this.padding_side === "right") {
              attention_mask.push(
                new Array(tokens[i].length).fill(1).concat(new Array(diff).fill(0))
              );
              tokens[i].push(...new Array(diff).fill(this.pad_token_id));
            } else {
              attention_mask.push(
                new Array(diff).fill(0).concat(new Array(tokens[i].length).fill(1))
              );
              tokens[i].unshift(...new Array(diff).fill(this.pad_token_id));
            }
          } else {
            attention_mask.push(new Array(tokens[i].length).fill(1));
          }
        }
      }
    } else {
      attention_mask = tokens.map((x) => new Array(x.length).fill(1));
    }
    if (return_tensor) {
      if (!(padding && truncation)) {
        if (tokens.some((x) => x.length !== tokens[0].length)) {
          throw Error(
            "Unable to create tensor, you should probably activate truncation and/or padding with 'padding=true' and 'truncation=true' to have batched tensors with the same length."
          );
        }
      }
      const dims = [tokens.length, tokens[0].length];
      if (return_tensor_dtype === "int32") {
        tokens = new Tensor4(
          return_tensor_dtype,
          Int32Array.from(tokens.flat()),
          dims
        );
        attention_mask = new Tensor4(
          return_tensor_dtype,
          Int32Array.from(attention_mask.flat()),
          dims
        );
      } else {
        tokens = new Tensor4(
          return_tensor_dtype,
          BigInt64Array.from(tokens.flat().map(BigInt)),
          dims
        );
        attention_mask = new Tensor4(
          return_tensor_dtype,
          BigInt64Array.from(attention_mask.flat().map(BigInt)),
          dims
        );
      }
    } else {
      if (!Array.isArray(text)) {
        tokens = tokens[0];
        attention_mask = attention_mask[0];
      }
    }
    let modelInputs = {
      input_ids: tokens,
      attention_mask
    };
    modelInputs = this.prepare_model_inputs(modelInputs);
    return modelInputs;
  }
  _encode_text(text) {
    if (text === null) {
      return [];
    }
    const sections = [...text.matchAll(this.added_tokens_regex)].map((x) => x[0]);
    return sections.map((x) => {
      if (this.added_tokens.includes(x)) {
        return x;
      } else {
        if (this.remove_space === true) {
          x = x.trim().split(/\s+/).join(" ");
        }
        if (this.normalizer !== null) {
          x = this.normalizer(x);
        }
        const sectionTokens = this.pre_tokenizer !== null ? this.pre_tokenizer(x) : [x];
        return this.model(sectionTokens);
      }
    }).flat();
  }
  static async from_pretrained(pretrained_model_name_or_path, options = { subdir: "tokenizer" }) {
    const [vocab, merges, tokenizerConfig] = await Promise.all([
      getModelJSON(pretrained_model_name_or_path, `${options.subdir}/vocab.json`, true, { revision: options.revision }),
      // getModelJSON(pretrained_model_name_or_path, `${options.subdir}/special_tokens_map.json`, true, options),
      getModelTextFile(pretrained_model_name_or_path, `${options.subdir}/merges.txt`, true, { revision: options.revision }),
      getModelJSON(pretrained_model_name_or_path, `${options.subdir}/tokenizer_config.json`, true, { revision: options.revision })
    ]);
    const tokenizerJSON = {
      normalizer: {
        type: "Lowercase"
      },
      pre_tokenizer: {
        type: "WhitespaceSplit"
      },
      post_processor: {
        type: "ByteLevel"
      },
      decoder: {
        type: "ByteLevel"
      },
      model: {
        type: "BPE",
        vocab,
        use_regex: true,
        end_of_word_suffix: "</w>",
        merges: merges.split("\n").slice(1, 49152 - 256 - 2 + 1)
      },
      added_tokens: []
    };
    return new _CLIPTokenizer(tokenizerJSON, tokenizerConfig);
  }
};

// src/pipelines/LCMStableDiffusionPipeline.ts
import { Tensor as Tensor6 } from "@xenova/transformers";

// src/pipelines/PipelineBase.ts
import { Tensor as Tensor5 } from "@xenova/transformers";
var PipelineBase = class {
  unet;
  vaeDecoder;
  vaeEncoder;
  textEncoder;
  tokenizer;
  scheduler;
  vaeScaleFactor;
  async encodePrompt(prompt) {
    const tokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: "int32"
      }
    );
    const inputIds = tokens.input_ids;
    const encoded = await this.textEncoder.run({ input_ids: new Tensor5("int32", Int32Array.from(inputIds.flat()), [1, inputIds.length]) });
    return encoded.last_hidden_state;
  }
  async getPromptEmbeds(prompt, negativePrompt) {
    const promptEmbeds = await this.encodePrompt(prompt);
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || "");
    return cat([negativePromptEmbeds, promptEmbeds]);
  }
  prepareLatents(batchSize, numChannels, height, width, seed = "") {
    const latentShape = [
      batchSize,
      numChannels,
      Math.floor(width / this.vaeScaleFactor),
      Math.floor(height / this.vaeScaleFactor)
    ];
    return randomNormalTensor(latentShape, void 0, void 0, "float32", seed);
  }
  async makeImages(latents) {
    latents = latents.div(this.vaeDecoder.config.scaling_factor || 0.18215);
    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents }
    );
    const images = decoded.sample.div(2).add(0.5).clipByValue(0, 1);
    return [images];
  }
  async release() {
    await this.unet?.release();
    await this.vaeDecoder?.release();
    await this.vaeEncoder?.release();
    await this.textEncoder?.release();
  }
};

// src/pipelines/LCMStableDiffusionPipeline.ts
var LCMStableDiffusionPipeline = class _LCMStableDiffusionPipeline extends PipelineBase {
  constructor(unet, vaeDecoder, vaeEncoder, textEncoder, tokenizer, scheduler) {
    super();
    this.unet = unet;
    this.vaeDecoder = vaeDecoder;
    this.vaeEncoder = vaeEncoder;
    this.textEncoder = textEncoder;
    this.tokenizer = tokenizer;
    this.scheduler = scheduler;
    this.vaeScaleFactor = 8;
  }
  static createScheduler(config) {
    return new LCMScheduler(
      {
        prediction_type: "epsilon",
        ...config
      }
    );
  }
  static async fromPretrained(modelRepoOrPath, inferMode, options) {
    const opts = {
      ...options
    };
    const usegpu = inferMode === "cpu" ? false : true;
    const unet = await loadModel(
      modelRepoOrPath,
      "unet/model.ort",
      opts,
      usegpu
    );
    const textEncoder = await loadModel(modelRepoOrPath, "text_encoder/model.ort", opts, usegpu);
    const vaeEncoder = await loadModel(modelRepoOrPath, "vae_encoder/model.ort", opts, usegpu);
    const vae = await loadModel(modelRepoOrPath, "vae_decoder/model.ort", opts, usegpu);
    const schedulerConfig = await getModelJSON(modelRepoOrPath, "scheduler/scheduler_config.json", true, opts);
    const scheduler = _LCMStableDiffusionPipeline.createScheduler(schedulerConfig);
    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: "tokenizer" });
    await dispatchProgress(opts.progressCallback, {
      status: "Ready" /* Ready */
    });
    return new _LCMStableDiffusionPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler);
  }
  getWEmbedding(batchSize, guidanceScale, embeddingDim = 512) {
    let w = new Tensor6("float32", new Float32Array([guidanceScale]), [1]);
    w = w.mul(1e3);
    const halfDim = embeddingDim / 2;
    let log = Math.log(1e4) / (halfDim - 1);
    let emb = range(0, halfDim).mul(-log).exp();
    emb = emb.mul(w.data[0]);
    return cat([emb.sin(), emb.cos()]).reshape([batchSize, embeddingDim]);
  }
  async run(input) {
    const width = input.width || 512;
    const height = input.height || 512;
    const batchSize = 1;
    const guidanceScale = input.guidanceScale || 7.5;
    const seed = input.seed || "";
    this.scheduler.setTimesteps(input.numInferenceSteps || 5);
    await dispatchProgress(input.progressCallback, {
      status: "EncodingPrompt" /* EncodingPrompt */
    });
    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt);
    let latents = this.prepareLatents(
      batchSize,
      this.unet.config.in_channels || 4,
      height,
      width,
      seed
    );
    let timesteps = this.scheduler.timesteps.data;
    const doClassifierFreeGuidance = guidanceScale > 1;
    let humanStep = 1;
    let cachedImages = null;
    const wEmbedding = this.getWEmbedding(batchSize, guidanceScale, 256);
    let denoised;
    for (const step of timesteps) {
      const timestep = input.sdV1 ? new Tensor6(BigInt64Array.from([BigInt(step)])) : new Tensor6(new Float32Array([step]));
      await dispatchProgress(input.progressCallback, {
        status: "RunningUnet" /* RunningUnet */,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length
      });
      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents;
      const noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds, timestep_cond: wEmbedding }
      );
      let noisePred = noise.out_sample;
      if (doClassifierFreeGuidance) {
        const [noisePredUncond, noisePredText] = [
          noisePred.slice([0, 1]),
          noisePred.slice([1, 2])
        ];
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale));
      }
      [latents, denoised] = this.scheduler.step(
        noisePred,
        step,
        latents
      );
      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: "RunningVae" /* RunningVae */,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length
        });
        cachedImages = await this.makeImages(denoised);
      }
      humanStep++;
    }
    await dispatchProgress(input.progressCallback, {
      status: "Done" /* Done */
    });
    if (input.runVaeOnEachStep) {
      return cachedImages;
    }
    return this.makeImages(latents);
  }
  async encodeImage(inputImage, width, height) {
    const encoded = await this.vaeEncoder.run(
      { sample: new Tensor6("float32", inputImage, [1, 3, width, height]) }
    );
    const encodedImage = encoded.latent_sample;
    return encodedImage.mul(0.18215);
  }
};

// src/pipelines/DiffusionPipeline.ts
var DiffusionPipeline = class {
  static async fromPretrained(modelRepoOrPath, options, device) {
    const opts = {
      ...options
    };
    const index = await getModelJSON(modelRepoOrPath, "model_index.json", true, opts);
    switch (index["_class_name"]) {
      case "ORTStableDiffusionPipeline":
        return LCMStableDiffusionPipeline.fromPretrained(modelRepoOrPath, device, options);
      default:
        throw new Error(`Unknown pipeline type ${index["_class_name"]}`);
    }
  }
};

// src/index.ts
setCacheImpl(browser_default);
export {
  DiffusionPipeline,
  LCMStableDiffusionPipeline,
  ProgressStatus,
  dispatchProgress,
  getModelFile,
  getModelJSON,
  getModelTextFile,
  loadModel,
  sessionRun,
  setCacheImpl,
  setModelCacheDir
};
