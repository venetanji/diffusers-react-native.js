// @ts-ignore
import * as ORT from 'onnxruntime-react-native'
import type { InferenceSession } from 'onnxruntime-react-native'
import { replaceTensors } from '../util/Tensor'
import { Tensor } from 'onnxruntime-react-native'

// @ts-ignore
const ONNX = ORT.default ?? ORT

const isNode = typeof process !== 'undefined' && process?.release?.name === 'node'

export class Session {
  private session: InferenceSession
  public config: Record<string, unknown>

  constructor (session: InferenceSession, config: Record<string, unknown> = {}, gpuEnable: boolean = false) {
    this.session = session
    this.config = config || {}
  }

  static async create (
    modelOrPath: string|ArrayBuffer,
    config?: Record<string, unknown>,
    gpuEnable?: boolean,
    options?: InferenceSession.SessionOptions,
  ) {
    const arg = typeof modelOrPath === 'string' ? modelOrPath : new Uint8Array(modelOrPath)

    console.log("Creating onnx session")
    // @ts-ignore
    const session = ONNX.InferenceSession.create(arg)

    // @ts-ignore
    return new Session(await session, config)
  }

  async run (inputs: Record<string, Tensor>) {
    // @ts-ignore
    const result = await this.session.run(inputs)
    return replaceTensors(result)
  }

  release () {
    // @ts-ignore
    return this.session.release()
  }
}
