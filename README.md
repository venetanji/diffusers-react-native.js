# diffuser.js based modification library for support ORTStableDiffusionPipeline With LCM Schedulers

original diffusers.js repo https://github.com/dakenf/diffusers.js

## Installation

```bash
npm i @waganawa/diffusers.js
```

## Usage

Browser (see examples/react)
```js
import { DiffusionPipeline } from '@waganawa/diffusers.js'

const pipe = DiffusionPipeline.fromPretrained('WGNW/chamcham_v1_checkpoint_onnx')
const images = pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 8,
})

const canvas = document.getElementById('canvas')
const data = await images[0].toImageData({ tensorLayout: 'NCWH', format: 'RGB' });
canvas.getContext('2d').putImageData(data, 0, 0);
```

Node.js (see examples/node)
```js
import { DiffusionPipeline } from '@waganawa/diffusers.js'
import { PNG } from 'pngjs'

const pipe = await DiffusionPipeline.fromPretrained('WGNW/chamcham_v1_checkpoint_onnx')
const images = await pipe.run({
  prompt: "an astronaut running a horse",
  numInferenceSteps: 8,
})

const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
p.data = Buffer.from(data.data)
p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
  console.log('Image saved as output.png');
})
```

'WGNW/chamcham_v1_checkpoint_onnx' is merge model of 'TechnoByte/MilkyWonderland' + LCM_LoRa model with my custom lora

## How does it work

It uses the original repo authors modified build of onnx runtime for web with 64bit and other changes. You can see the detail list of contributions here https://github.com/dakenf/diffusers.js?tab=readme-ov-file#how-does-it-work
