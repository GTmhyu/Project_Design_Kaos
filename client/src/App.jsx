import Canvas from "./canvas";
import Customizer from "./pages/Customizer";
import Home from "./pages/Home";
import Chat from "./components/Chat-bot";

function App() {

  return (
    <main className="app transition-all ease-in">
      
      <Chat />
      <Home />
      <Canvas />
      <Customizer />

    </main>
  )
}

export default App
