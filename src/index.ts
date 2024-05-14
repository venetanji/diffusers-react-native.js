// import 'module-alias/register.js'
import browserCache from './hub/browser'
import { setCacheImpl } from './hub'

export * from './pipelines/LCMStableDiffusionPipeline'
export * from './pipelines/DiffusionPipeline'
export * from './pipelines/common'
export * from './hub'
export { setModelCacheDir } from './hub/browser'

setCacheImpl(browserCache)
