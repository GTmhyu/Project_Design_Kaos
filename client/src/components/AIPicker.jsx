import React from 'react'

import CustomButton from './CustomButton';

const AIPicker = ({ prompt, setPrompt, generatingImg, handleSubmit }) => {
  return (
    <div className="aipicker-container">
      <textarea 
        placeholder="Ask AI..."
        rows={5}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        className="aipicker-textarea"
      />
      <div className='flex flex-wrap gap-2'>
        {generatingImg ? (
          <CustomButton
          type='outline'
          tittle='Ask me'
          customStyles="text-xs"
          />
        ): (
          <>
            <CustomButton
            type='outline'
            tittle='Logo'
            customStyles="text-xs"
            handleClick={() => handleSubmit('logo')}
            />
            <CustomButton 
            type='filled'
            tittle='Kaos' 
            handleClick={() => handleSubmit('full')}
            customStyles="text-xs"
            />
          </>
        ) }
      </div>
    </div>
  )
}

export default AIPicker