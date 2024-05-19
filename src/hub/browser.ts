import { downloadFile } from '@huggingface/hub'
import { DbCache } from './indexed-db'
import { GetModelFileOptions, pathJoin } from './common'
import { dispatchProgress, ProgressCallback, ProgressStatus } from '../pipelines/common'
import * as RNFS from 'react-native-fs'

let cacheDir = ''
export function setModelCacheDir (dir: string) {
  cacheDir = dir
}

export function getCacheKey (modelRepoOrPath: string, fileName: string, revision: string) {
  return pathJoin(cacheDir, modelRepoOrPath, revision === 'main' ? '' : revision, fileName)
}

export async function getModelFile (modelRepoOrPath: string, fileName: string, fatal = true, options: GetModelFileOptions = {}) {
  const revision = options.revision || 'main'
  // const cachePath = getCacheKey(modelRepoOrPath, fileName, revision)
  // const cache = new DbCache()
  // await cache.init()
  // const cachedData = await cache.retrieveFile(cachePath, options.progressCallback, fileName)
  // if (cachedData) {
  //   if (options.returnText) {
  //     const decoder = new TextDecoder('utf-8')
  //     return decoder.decode(cachedData.file)
  //   }

  //   return cachedData.file
  // }

  let response: Response|null|undefined
  // now local cache
  // if (cacheDir) {
  //   response = await fetch(cachePath)
  //   // create-react-app will return 200 with HTML for missing files
  //   if (!response || !response.body || response.status !== 200 || response.headers.get('content-type')?.startsWith('text/html')) {
  //     response = null
  //   }
  // }

  try {
    // now try the hub
    const folder = "/storage/emulated/0/Android/data/com.playgen/files"
    const alreadyDownloaded = await RNFS.exists(`file://${folder}/${fileName}`)
    console.log(alreadyDownloaded)
    if (!alreadyDownloaded) {
      console.log('Downloading', modelRepoOrPath, fileName, revision)
      const response = await RNFS.downloadFile({
        fromUrl: `https://huggingface.co/${modelRepoOrPath}/resolve/${revision}/${fileName}?download=true`,
        toFile: `file://${folder}/${fileName}`,
      }).promise
      console.log(response)
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

function readResponseToBuffer (response: Response, progressCallback: ProgressCallback, displayName: string): Promise<ArrayBuffer> {
  const contentLength = response.headers.get('content-length')
  if (!contentLength) {
    return response.arrayBuffer()
  }

  let buffer: ArrayBuffer
  const contentLengthNum = parseInt(contentLength, 10)

  if (contentLengthNum > 2 * 1024 * 1024 * 1024) {
    // @ts-ignore
    const memory = new WebAssembly.Memory({ initial: Math.ceil(contentLengthNum / 65536), index: 'i64' })
    buffer = memory.buffer
  } else {
    buffer = new ArrayBuffer(contentLengthNum)
  }

  let offset = 0
  return new Promise((resolve, reject) => {
    const reader = response.body!.getReader()

    async function pump (): Promise<void> {
      const { done, value } = await reader.read()
      if (done) {
        return resolve(buffer)
      }
      const chunk = new Uint8Array(buffer, offset, value.byteLength)
      chunk.set(new Uint8Array(value))
      offset += value.byteLength
      await dispatchProgress(progressCallback, {
        status: ProgressStatus.Downloading,
        downloadStatus: {
          file: displayName,
          size: contentLengthNum,
          downloaded: offset,
        }
      })
      return pump()
    }

    pump().catch(reject)
  })
}

export default {
  getModelFile,
}
