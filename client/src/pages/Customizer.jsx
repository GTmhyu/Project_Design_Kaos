import React, { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useSnapshot } from 'valtio';

import config from '../config/config';
import state from '../store';
import { download } from '../assets';
import { downloadCanvasToImage, postImage, reader } from '../config/helpers';
import { EditorTabs, FilterTabs, DecalTypes } from '../config/constants';
import { fadeAnimation, slideAnimation } from '../config/motion';
import { AIPicker, ColorPicker, Tab, FilePicker, CustomButton } from '../components';
import { Torus } from '@react-three/drei';

const Coustomizer = () => {
  const snap = useSnapshot(state);

  const [file, setFile] = useState('');
  const [prompt, setPrompt] = useState('');
  const [generatingImg, setGeneratingImg] = useState(false);

  const [activeEditorTab, setactiveEditorTab] = useState("")
  const [activeFilterTab, setActiveFilterTab] = useState({
    logoShirt: true,
    stylishShirt: false,
  });

  //show tab content depending on activeTab
  const generateTabContent = () => {
    switch (activeEditorTab) {
      case "colorpicker":
        return <ColorPicker />
      case "filepicker":
        return <FilePicker 
        file={file}
        setFile={setFile}
        readFile={readfile}
        />
      case "aipicker":
        return <AIPicker />
      default:
        return null;

    }
  }
  const handleDecals = (type, result) => {
     const decalTypes = DecalTypes[type];

     state[decalTypes.stateProperty] = result;
     if (!activeFilterTab[decalTypes.isFilterTab]) {
      handleActiveFilterTab(decalTypes.isFilterTab);
     }
  }

  const handleActiveFilterTab = (tabname) => {
    switch (tabname) {
      case "logoShirt":
        state.isLogoTexture = !activeFilterTab[tabname];
        break;
      case "stylishShirt":
        state.isFullTexture = !activeFilterTab[tabname];
        break;
      default:
        state.isLogoTexture = true;
        state.isFullTexture = false;

    }
  }

  const readfile = (type) => {
    reader(file).then((result) => {
      handleDecals(result, type);
      setactiveEditorTab('');
    })
  }

  return (
    <AnimatePresence>
      {!snap.intro && (
        <>
        <motion.div 
         key="Custom"
         className='absolute top-0 left-0 z-10'
         {...slideAnimation('left')}

        >
          <div className='flex items-center min-h-screen'>
            <div className='editortabs-container tabs'>
            {EditorTabs.map((tab) => (
              <Tab 
                key={tab.name}
                tab={tab}
                handleClick= {() => setactiveEditorTab(tab.name)}
              />
            ))}
            {generateTabContent()}
            </div>  
          </div>
        </motion.div>

        <motion.div 
        className='absolute z-10 top-5 right-5'
        {...fadeAnimation}
        >
          <CustomButton 
            type='filled'
            tittle="Back"
            handleClick={() => state.intro = true}
            customStyles="w-fit px-4 py-2.5 font-bold text-sm"
          />

        </motion.div >

          <motion.div
            className='filtertabs-container'
            {...slideAnimation('up')}
          >
            {FilterTabs.map((tab) => (
              <Tab
                key={tab.name}
                tab={tab}
                isFilterTab
                isActiveTab=""
                handleClick={() => { }}
              />
            ))}
            <button className='download-btn' onClick={downloadCanvasToImage}>
              <img
                src={download}
                alt='download_image'
                className='w-3/5 h-3/5 object-contain'
              />
            </button>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

export default Coustomizer