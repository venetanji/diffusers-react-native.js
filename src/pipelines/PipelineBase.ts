import { Session } from '../backends'
import { CLIPTokenizer } from '../tokenizers/CLIPTokenizer'
import { SchedulerBase } from '../schedulers/SchedulerBase'
import { Tensor } from '@xenova/transformers'
import {cat, randomNormalTensor} from '../util/Tensor'

export class PipelineBase {
  public unet: Session
  public vaeDecoder: Session
  public vaeEncoder: Session
  public textEncoder: Session
  public tokenizer: CLIPTokenizer
  public scheduler: SchedulerBase
  public vaeScaleFactor: number

  async encodePrompt (prompt: string): Promise<Tensor> {
    const tokens = this.tokenizer(
      prompt,
      {
        return_tensor: false,
        padding: true,
        max_length: this.tokenizer.model_max_length,
        return_tensor_dtype: 'int32',
      },
    )

    const inputIds = tokens.input_ids
    // @ts-ignore
    const encoded = await this.textEncoder.run({ input_ids: new Tensor('int32', Int32Array.from(inputIds.flat()), [1, inputIds.length]) })
    return encoded.last_hidden_state
  }

  async getPromptEmbeds (prompt: string, negativePrompt: string | undefined) {
    const promptEmbeds = await this.encodePrompt(prompt)
    const negativePromptEmbeds = await this.encodePrompt(negativePrompt || '')

    return cat([negativePromptEmbeds, promptEmbeds])
  }

  prepareLatents (batchSize: number, numChannels: number, height: number, width: number, seed = '') {
    const latentShape = [
      batchSize,
      numChannels,
      Math.floor(width / this.vaeScaleFactor),
      Math.floor(height / this.vaeScaleFactor),
    ]

    return randomNormalTensor(latentShape, undefined, undefined, 'float32', seed)
  }

  async makeImages (latents: Tensor) {
    console.log("VAE")
    // using taesd scaling is 1 so this is commented out
    //latents = latents.div(this.vaeDecoder.config.scaling_factor || 0.18215)

    const decoded = await this.vaeDecoder.run(
      { latent_sample: latents },
    )
    
    const raw = decoded.sample
      .div(2)
      .add(0.5)
      .mul(255)
      .round()
      .clipByValue(0, 255)

    const image = raw[0]

    const alpha = new Tensor('float32', new Float32Array(raw.dims[2] * raw.dims[3]).fill(255), [1, raw.dims[2], raw.dims[3]])
    const alphaimage = cat([image, alpha], 0)
    const images = alphaimage.transpose(1, 2, 0)

    console.log("VAE Done")
    return [images]
  }

  async release () {
    await this.unet?.release()
    await this.vaeDecoder?.release()
    await this.vaeEncoder?.release()
    await this.textEncoder?.release()
  }
}
