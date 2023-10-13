import React from 'react'
import state from '../store'
import {useSnapshot} from 'valtio'

const CustomButton = ({type, tittle, customStyles, handleClick}) => {
    const snap = useSnapshot(state);

    const generateStyles = (type) => {
        if(type === 'filled') 
        return ({
            backgroundColor: snap.color,
            color: '#fff'
        })
    }

  return (
    <button 
    className={`px-2 py-1.5 rounded-md ${customStyles}`}
    style={generateStyles(type)}
    onClick={handleClick}
    >
        {tittle}
    </button>
  )
}

export default CustomButton