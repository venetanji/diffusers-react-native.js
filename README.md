# diffuser.js for react-native

Original diffusers.js repo https://github.com/dakenf/diffusers.js

Only LCM is working at the moment. Adapted to use tased as vae.

## Installation

Add the dependency to your react native project

```bash
yarn add https://github.com/venetanji/diffusers-react-native.js
```

Alias transformers.js in package.json:

```
"@xenova/transformers": "https://github.com/venetanji/transformers.js.git#merge",
```
Forked from: https://github.com/hans00/transformers.js/tree/merge

## Usage

Example App.tsx:

```
import React, { useEffect, useState, useRef } from 'react';
import type {PropsWithChildren} from 'react';
import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  useColorScheme,
  View,
  Button,
  Alert,
  Image
} from 'react-native';
import { LCMStableDiffusionPipeline } from '@venetanji/diffusers.js'
import { Skia, AlphaType, ColorType } from "@shopify/react-native-skia";


const App = () => {
  const [pipe, setPipe] = useState(null);
  const createSession = async () => {
    // download model if the file does not exist
    if (pipe === null) {

      const apipe = await LCMStableDiffusionPipeline.fromPretrained('Aryanne/tiny_sd_lcm_ort')
      setPipe(apipe)

    } else {
      console.log('Session already created')
    }
  };

  const [imageUri, setImageUri] = useState({uri: 'https://reactnative.dev/img/tiny_logo.png'});


  const generate = async () => {
    if (pipe === null) {
      console.log('Session not created')
      return
    }
    //await createSession()
    console.log("Generating image...")
    const square = 512
    const {width, height} = {width: square, height: square}

    const progressCallback = (payload:any) => {
      console.log(payload)
    }

    const images = await pipe.run({
      prompt: "A photograph of a horse on mars, highly detailed",
      width: width,
      height: height,
      numInferenceSteps: 4,
      guidanceScale: 1.8,
      //beta_schedule: "scaled_linear",
      progressCallback: progressCallback
    })
    const pixels = new Uint8Array(images[0].data);

    const imagedata = Skia.Data.fromBytes(pixels)
    // console.log("Creating skia image")
    const img = Skia.Image.MakeImage(
      {
        width: width,
        height: height,
        colorType: ColorType.RGBA_8888,
        alphaType: AlphaType.Premul,
      },
      imagedata,
      width * 4
    );
    if (!img) {
      console.error('Failed to create image');
      return;
    }

    setImageUri({uri: `data:image/png;base64,${img?.encodeToBase64()}`})
    console.log('Image encoded and sent to Image component')
  }

  const GeneratedImage = () => {

    return (
      <Image
        style={{width: 400, height: 400}}
        source={imageUri}
      />
    );
  };

  return (
    <View style={styles.container}>
      <Text>React native diffusion</Text>
      <GeneratedImage></GeneratedImage>
      <Button
        title="Create Session"
        onPress={() => {
          createSession();
        }}
        ></Button>
      <Button
        title="Generate"
        onPress={() => {
          generate();
        }}
        ></Button>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'space-around',
  },
  btn: { width: '90%' },
});
export default App;
```




