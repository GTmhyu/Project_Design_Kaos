import { proxy } from "valtio";

const state = proxy({

  intro: true,
  color:'#EFBD48',
  isLogoTexture: true,
  isFullTexture: false,
  logoDecal : './Icozy_T.png',
  fullDecal : './threejs.png',
  
});

export default state;