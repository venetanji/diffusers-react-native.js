"use strict"
import { LCMScheduler, LCMSchedulerConfig } from '@/schedulers/LCMScheduler'
import { CLIPTokenizer } from '@/tokenizers/CLIPTokenizer'
import { cat, randomNormalTensor, range } from '@/util/Tensor'
import { Tensor } from '@xenova/transformers'
import { dispatchProgress, loadModel, PretrainedOptions, ProgressCallback, ProgressStatus } from './common'
import { getModelJSON } from '@/hub'
import { Session } from '@/backends'
import { GetModelFileOptions } from '@/hub/common'
import { PipelineBase } from '@/pipelines/PipelineBase'

export interface StableDiffusionInput {
  prompt: string
  negativePrompt?: string
  guidanceScale?: number
  seed?: string
  width?: number
  height?: number
  numInferenceSteps: number
  sdV1?: boolean
  progressCallback?: ProgressCallback
  runVaeOnEachStep?: boolean
  img2imgFlag?: boolean
  inputImage?: Float32Array
  strength?: number
}

export class StableDiffusionPipeline extends PipelineBase {
  declare scheduler: LCMScheduler

  constructor (unet: Session, vaeDecoder: Session, vaeEncoder: Session, textEncoder: Session, tokenizer: CLIPTokenizer, scheduler: LCMScheduler) {
    super()
    this.unet = unet
    this.vaeDecoder = vaeDecoder
    this.vaeEncoder = vaeEncoder
    this.textEncoder = textEncoder
    this.tokenizer = tokenizer
    this.scheduler = scheduler
    this.vaeScaleFactor = 8
  }

  static createScheduler (config: LCMSchedulerConfig) {
    return new LCMScheduler(
      {
        prediction_type: 'epsilon',
        ...config,
      },
    )
  }

  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    // order matters because WASM memory cannot be decreased. so we load the biggest one first
    const unet = await loadModel(
      modelRepoOrPath,
      'unet/model.onnx',
      opts,
    )
    const textEncoder = await loadModel(modelRepoOrPath, 'text_encoder/model.onnx', opts)
    const vaeEncoder = await loadModel(modelRepoOrPath, 'vae_encoder/model.onnx', opts)
    const vae = await loadModel(modelRepoOrPath, 'vae_decoder/model.onnx', opts)

    const schedulerConfig = await getModelJSON(modelRepoOrPath, 'scheduler/scheduler_config.json', true, opts)
    const scheduler = StableDiffusionPipeline.createScheduler(schedulerConfig)

    const tokenizer = await CLIPTokenizer.from_pretrained(modelRepoOrPath, { ...opts, subdir: 'tokenizer' })
    await dispatchProgress(opts.progressCallback, {
      status: ProgressStatus.Ready,
    })
    return new StableDiffusionPipeline(unet, vae, vaeEncoder, textEncoder, tokenizer, scheduler)
  }

  getWEmbedding (batchSize: number, guidanceScale: number, embeddingDim = 512) {
    let w = new Tensor('float32', new Float32Array([guidanceScale]), [1])
    w = w.mul(1000)

    const halfDim = embeddingDim / 2
    let log = Math.log(10000) / (halfDim - 1)
    let emb: Tensor = range(0, halfDim).mul(-log).exp()

    // TODO: support batch size > 1
    emb = emb.mul(w.data[0])

    return cat([emb.sin(), emb.cos()]).reshape([batchSize, embeddingDim])
  }

  async run (input: StableDiffusionInput) {
    const width = input.width || 512
    const height = input.height || 512
    const batchSize = 1
    const guidanceScale = input.guidanceScale || 7.5
    const seed = input.seed || ''
    this.scheduler.setTimesteps(input.numInferenceSteps || 5)

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.EncodingPrompt,
    })

    const promptEmbeds = await this.getPromptEmbeds(input.prompt, input.negativePrompt)

    let latents = this.prepareLatents(
      batchSize,
      this.unet.config.in_channels as number || 4,
      height,
      width,
      seed,
    ) // Normal latents used in Text-to-Image
    let timesteps = this.scheduler.timesteps.data

    const doClassifierFreeGuidance = guidanceScale > 1
    let humanStep = 1
    let cachedImages: Tensor[] | null = null

    const wEmbedding = this.getWEmbedding(batchSize, guidanceScale, 256)
    let denoised: Tensor

    for (const step of timesteps) {
      // for some reason v1.4 takes int64 as timestep input. ideally we should get input dtype from the model
      // but currently onnxruntime-node does not give out types, only input names
      const timestep = input.sdV1
        ? new Tensor(BigInt64Array.from([BigInt(step)]))
        : new Tensor(new Float32Array([step]))
      await dispatchProgress(input.progressCallback, {
        status: ProgressStatus.RunningUnet,
        unetTimestep: humanStep,
        unetTotalSteps: timesteps.length,
      })

      const latentInput = doClassifierFreeGuidance ? cat([latents, latents.clone()]) : latents

      const noise = await this.unet.run(
        { sample: latentInput, timestep, encoder_hidden_states: promptEmbeds, timestep_cond: wEmbedding },
      )

      let noisePred = noise.out_sample
      if (doClassifierFreeGuidance) {
        const [noisePredUncond, noisePredText] = [
          noisePred.slice([0, 1]),
          noisePred.slice([1, 2]),
        ]
        noisePred = noisePredUncond.add(noisePredText.sub(noisePredUncond).mul(guidanceScale))
      }

      [latents, denoised] = this.scheduler.step(
        noisePred,
        step,
        latents,
      )

      if (input.runVaeOnEachStep) {
        await dispatchProgress(input.progressCallback, {
          status: ProgressStatus.RunningVae,
          unetTimestep: humanStep,
          unetTotalSteps: timesteps.length,
        })
        cachedImages = await this.makeImages(denoised)
      }
      humanStep++
    }

    await dispatchProgress(input.progressCallback, {
      status: ProgressStatus.Done,
    })

    if (input.runVaeOnEachStep) {
      return cachedImages!
    }

    return this.makeImages(latents)
  }

  async encodeImage (inputImage: Float32Array, width: number, height: number) {
    const encoded = await this.vaeEncoder.run(
      { sample: new Tensor('float32', inputImage, [1, 3, width, height]) },
    )

    const encodedImage = encoded.latent_sample
    return encodedImage.mul(0.18215)
  }
}
