import React, { useState, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { useSnapshot } from 'valtio';

import { signInWithGoogle } from '../firebase.js';
import config from '../config/config';
import state from '../store';
import { feedback } from '../assets';
import { chatbot } from '../assets';
import { download } from '../assets';
import { downloadCanvasToImage, reader } from '../config/helpers';
import { EditorTabs, FilterTabs, DecalTypes } from '../config/constants';
import { fadeAnimation, slideAnimation } from '../config/motion';
import { AIPicker, ColorPicker,Tab, FilePicker, CustomButton } from '../components';
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
        readFile={readFile}
        />
      case "aipicker":
        return <AIPicker 
        prompt={prompt}
        setPrompt={setPrompt}
        generatingImg={generatingImg}
        handleSubmit={handleSubmit}
        />
      default:
        return null;

    }
  }

  const handleSubmit = async (type) => {
    if(!prompt) return alert("Masukan Kalimat terlebih dulu!");

    try {
      setGeneratingImg(true);

      const response = await fetch('http://localhost:8080/api/v1/dalle', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          prompt,
        })
      })

      const data = await response.json();

      handleDecals(type, `data:image/png;base64,${data.photo}`)
    } catch (error) {
      alert(error)
    } finally {
      setGeneratingImg(false);
      setActiveEditorTab("");
    }
  }

  const handleDecals = (type, result) => {
    const decalType = DecalTypes[type];

    state[decalType.stateProperty] = result;

    if(!activeFilterTab[decalType.filterTab]) {
      handleActiveFilterTab(decalType.filterTab)
    }
  }

  const handleActiveFilterTab = (tabName) => {
    switch (tabName) {
      case "logoShirt":
          state.isLogoTexture = !activeFilterTab[tabName];
        break;
      case "stylishShirt":
          state.isFullTexture = !activeFilterTab[tabName];
        break;
      default:
        state.isLogoTexture = true;
        state.isFullTexture = false;
        break;
    }

    // after setting the state, activeFilterTab is updated

    setActiveFilterTab((prevState) => {
      return {
        ...prevState,
        [tabName]: !prevState[tabName]
      }
    })
  }

  const readFile = (type) => {
    reader(file)
      .then((result) => {
        handleDecals(type, result);
        setActiveEditorTab("");
      })
  }


  return (
    <AnimatePresence>
      {!snap.intro && (
        <><>
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
                    handleClick={() => setactiveEditorTab(tab.name)} />
                ))}
                {generateTabContent()}
              </div>
            </div>
            
          </motion.div>

          <motion.div
            className='absolute z-10 top-5 right-8'
            {...fadeAnimation}
          >
            <CustomButton
              type='filled'
              tittle="Back"
              handleClick={() => state.intro = true}
              customStyles="w-fit px-4 py-2.5 font-bold text-sm" />

          </motion.div>
          <motion.div
            className='absolute z-10 top-5 right-28'
            {...fadeAnimation}
          >
            <CustomButton
            type='filled'
            tittle="Login"
            handleClick={signInWithGoogle}
            customStyles="w-fit px-4 py-2.5 font-bold text-sm"
            />

          </motion.div>

          <motion.div
            className='filtertabs-container'
            {...slideAnimation('up')}
          >
            {FilterTabs.map((tab) => (
              <Tab
                key={tab.name}
                tab={tab}
                isFilterTab
                isActiveTab={activeFilterTab[tab.name]}
                handleClick={() => handleActiveFilterTab(tab.name)} />
            ))}
            <button className='download-btn' onClick={downloadCanvasToImage}>
              <img
                src={download}
                alt='download_image'
                className='w-3/5 h-3/5 object-contain' />
            </button>
          </motion.div>

        </><motion.div
          key="Custom"
          className='absolute top-0 right-0 z-10'
          {...slideAnimation('right')}
        >
          <div className='flex items-center min-h-screen'>
            <div className='editortabs-container tabs'>
              
              <a className='feedback-btn' href='http://127.0.0.1:5000/#'>
                <img 
                src={feedback}
                alt='feedback'
                className='w-4/5 h-3/5 object-contain'
                />
              </a>
              <a className='chatbot-btn' href='http://localhost:7000'>
                <img 
                src={chatbot}
                alt='chatbot'
                className='w-4/5 h-3/5 object-contain'
                />
              </a>
            </div>
          </div>
          </motion.div></>
      )}

    </AnimatePresence>
  )
}

export default Coustomizer

