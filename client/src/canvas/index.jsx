import { Canvas } from "@react-three/fiber";
import { Center, Environment } from "@react-three/drei";
import Shirt from "./Shirt";
import CameraRig from "./CameraRig";
import Background from "./Background";


const CanvasModel = () => {
  return (
    <Canvas>
          <ambientLight intensity={0.5} />
    <Environment preset="city" />

    {/*<CameraRig>*/}
      {/*<Background/>*/}
      <Center>
      <Shirt/>
      </Center>
    {/*</CameraRig>*/}
    </Canvas>
  )
}

export default CanvasModel