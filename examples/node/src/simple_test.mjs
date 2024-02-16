import { DiffusionPipeline } from '@waganawa/diffusers.js'
import { PNG } from 'pngjs'
import fs from 'fs'

function main() {
  const pipe = await DiffusionPipeline.fromPretrained('WGNW/chamcham_LCM_v1_onnx_fp16',{
    revision: "main",
  })
  
  const images = await pipe.run({
    prompt: "sharp details, sharp focus, anime style, masterpiece, best quality, chamcham(twitch), hair bell, hair ribbon, multicolored hair, two-tone hair, 1girl, solo, orange shirt, long hair, hair clip",
    negativePrompt: "bad hand,text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated text,watermark,low quality,medium quality,blurry,censored,wrinkles,deformed,mutated",
    numInferenceSteps: 8,
    sdV1: true,
    height: 512,
    width: 512,
    guidanceScale: 2,
    img2imgFlag: false,
    runVaeOnEachStep: true,
  })
  
  const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)
  
  const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
  p.data = Buffer.from(data.data)
  p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
    console.log('Image saved as output.png');
  })
}