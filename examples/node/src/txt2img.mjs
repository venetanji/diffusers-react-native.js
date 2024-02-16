import minimist from 'minimist';
import { DiffusionPipeline } from '@waganawa/diffusers.js'
import fs from 'fs'
import { PNG } from 'pngjs'
import progress from 'cli-progress'

function parseCommandLineArgs() {
  const args = minimist(process.argv.slice(2));

  return {
    m: args.m || 'aislamov/stable-diffusion-2-1-base-onnx',
    prompt: args.prompt || 'an astronaut riding a horse',
    negativePrompt: args.negativePrompt || '',
    rev: args.rev,
    version: args.version || 1,
    steps: args.steps || 8,
  }
}

async function main() {
  const args = parseCommandLineArgs();
  const pipe = await DiffusionPipeline.fromPretrained(
    args.m,
    {
      revision: args.rev,
    }
  )

  const progressBar = new progress.SingleBar({
  }, progress.Presets.shades_classic)

  progressBar.start(args.steps + 1, 0)

  const images = await pipe.run({
    prompt: args.prompt,
    negativePrompt: args.negativePrompt,
    numInferenceSteps: args.steps,
    sdV1: args.version === 1,
    height: 512,
    width: 512,
    guidanceScale: 2,
    img2imgFlag: false,
    runVaeOnEachStep: true,
    progressCallback: (progress) => {
      progressBar.update(progress.unetTimestep)
    },
  })
  progressBar.stop()
  const data = await images[0].mul(255).round().clipByValue(0, 255).transpose(0, 2, 3, 1)

  const p = new PNG({ width: 512, height: 512, inputColorType: 2 })
  p.data = Buffer.from(data.data)
  p.pack().pipe(fs.createWriteStream('output.png')).on('finish', () => {
    console.log('Image saved as output.png');
  })
}

main();
