import { initializeApp } from "firebase/app";
import { getAuth,GoogleAuthProvider, signInWithPopup } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCwdJV1ulPJ2jIrdPkXMtfNHsN4GPzT3ng",
  authDomain: "signin-example-a3c39.firebaseapp.com",
  projectId: "signin-example-a3c39",
  storageBucket: "signin-example-a3c39.appspot.com",
  messagingSenderId: "657399172326",
  appId: "1:657399172326:web:c225a3708182ffcea67f27",
  measurementId: "G-PSTSB8LX94"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);

const provider = new GoogleAuthProvider();

export const signInWithGoogle = () => {
    signInWithPopup(auth, provider).then((result) => {
        console.log(result);
    })
    .catch((error) => {
        console.log(error);
    })
};