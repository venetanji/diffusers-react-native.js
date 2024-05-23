import { GetModelFileOptions, pathJoin } from './common'
import { dispatchProgress, ProgressCallback, ProgressStatus } from '../pipelines/common'
import * as RNFS from 'react-native-fs'

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const revision = options.revision || 'main'

  try {
    const folder = RNFS.CachesDirectoryPath
    // get the parent folder in filename
    const alreadyDownloaded = await RNFS.exists(`file://${folder}/${fileName}`)
    //console.log(alreadyDownloaded)
    if (!alreadyDownloaded) {
      const lastSlash = fileName.lastIndexOf('/')
      if (lastSlash > 0) {
        const parentFolder = fileName.substring(0, lastSlash)
        await RNFS.mkdir(`${folder}/${parentFolder}`)
      }
      //console.log('Downloading', modelRepoOrPath, fileName, revision)
      const response = await RNFS.downloadFile({
        fromUrl: `https://huggingface.co/${modelRepoOrPath}/resolve/${revision}/${fileName}?download=true`,
        toFile: `file://${folder}/${fileName}`,
      }).promise
      //console.log(response)
    }
    if (options.returnText) {
      return await RNFS.readFile(folder + '/' + fileName, 'utf8')
    } else {
      return `${folder}/${fileName}`
    }
  } catch (e) {
    if (!fatal) {
      return null
    }
    throw e
  }
}

export default {
  getModelFile,
}
