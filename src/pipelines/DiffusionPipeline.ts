import { PretrainedOptions } from '@/pipelines/common'
import { GetModelFileOptions } from '@/hub/common'
import { getModelJSON } from '../hub'
import { LCMStableDiffusionPipeline } from './LCMStableDiffusionPipeline'

export class DiffusionPipeline {
  static async fromPretrained(modelRepoOrPath: string, options?: PretrainedOptions, device?: string) {
    const opts: GetModelFileOptions = {
      ...options,
    }

    const index = await getModelJSON(modelRepoOrPath, 'model_index.json', true, opts)
    switch (index['_class_name']) {
      case 'ORTStableDiffusionPipeline':
        return LCMStableDiffusionPipeline.fromPretrained(modelRepoOrPath, device, options)
      default:
        throw new Error(`Unknown pipeline type ${index['_class_name']}`)
    }
  }
}
