import { Canvas } from "@react-three/fiber";
import { Center, Environment } from "@react-three/drei";
import Shirt from "./Shirt";
import CameraRig from "./CameraRig";
import Background from "./Background";


const CanvasModel = () => {
  return (
    <Canvas 
      shadows
      // Ubah ukuran Kaos fov
      camera={{ position: [0, 0, 1.5], fov: 43 }}
      gl={{ preserveDrawingBuffer: true }}
      className="w-full mas-w-full h-full transition-all ease-in"
    >
          <ambientLight intensity={0.5} />
    <Environment preset="city" />

    <CameraRig>
      <Background/>
      <Center>
      <Shirt/>
      </Center>
    </CameraRig>
    </Canvas>
  )
}

export default CanvasModel