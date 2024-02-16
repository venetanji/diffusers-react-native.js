import { PretrainedOptions } from '@/pipelines/common'
import { GetModelFileOptions } from '@/hub/common'
import { getModelJSON } from '@/hub'
import { StableDiffusionPipeline } from '@/pipelines/StableDiffusionPipeline'
import { LCMScheduler } from '@/schedulers/LCMScheduler'

export class DiffusionPipeline {
  static async fromPretrained (modelRepoOrPath: string, options?: PretrainedOptions) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const index = await getModelJSON(modelRepoOrPath, 'model_index.json', true, opts)
    let pipe: StableDiffusionPipeline
    switch (index['_class_name']) {
      case 'ORTStableDiffusionPipeline':
        return StableDiffusionPipeline.fromPretrained(modelRepoOrPath, options)
      default:
        throw new Error(`Unknown pipeline type ${index['_class_name']}`)
    }
  }
}
