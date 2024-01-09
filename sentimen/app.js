import { pipeline } from "@xenova/transformers";
import express from "express";
import {fileURLToPath} from 'url';
import path from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const pipe = await pipeline('sentiment-analysis'); 

const app = express();

app.use(express.json());

app.use('/public', express.static('./public'));

app.post('/', async (req, res) => {
    const result = await pipe(req.body.text);
    res.json(result);
})

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
})

app.listen(3000, () => { 
    console.log('Server listening on port: 3000');
})