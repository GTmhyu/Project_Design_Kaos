declare const FeatureExtractor_base: new () => {
    (...args: any[]): any;
    _call(...args: any[]): any;
};
/**
 * Named tuple to indicate the order we are using is (height x width), even though
 * the Graphics’ industry standard is (width x height).
 * @typedef {[height: number, width: number]} HeightWidth
 */
/**
 * Base class for feature extractors.
 *
 * @extends Callable
 */
export class FeatureExtractor extends FeatureExtractor_base {
    /**
     * Constructs a new FeatureExtractor instance.
     *
     * @param {Object} config The configuration for the feature extractor.
     */
    constructor(config: any);
    config: any;
}
/**
 * @typedef {object} ImageFeatureExtractorResult
 * @property {Tensor} pixel_values The pixel values of the batched preprocessed images.
 * @property {HeightWidth[]} original_sizes Array of two-dimensional tuples like [[480, 640]].
 * @property {HeightWidth[]} reshaped_input_sizes Array of two-dimensional tuples like [[1000, 1330]].
 */
/**
 * Feature extractor for image models.
 *
 * @extends FeatureExtractor
 */
export class ImageFeatureExtractor extends FeatureExtractor {
    /**
     * Constructs a new ImageFeatureExtractor instance.
     *
     * @param {Object} config The configuration for the feature extractor.
     * @param {number[]} config.image_mean The mean values for image normalization.
     * @param {number[]} config.image_std The standard deviation values for image normalization.
     * @param {boolean} config.do_rescale Whether to rescale the image pixel values to the [0,1] range.
     * @param {number} config.rescale_factor The factor to use for rescaling the image pixel values.
     * @param {boolean} config.do_normalize Whether to normalize the image pixel values.
     * @param {boolean} config.do_resize Whether to resize the image.
     * @param {number} config.resample What method to use for resampling.
     * @param {number} config.size The size to resize the image to.
     */
    constructor(config: {
        image_mean: number[];
        image_std: number[];
        do_rescale: boolean;
        rescale_factor: number;
        do_normalize: boolean;
        do_resize: boolean;
        resample: number;
        size: number;
    });
    image_mean: any;
    image_std: any;
    resample: any;
    do_rescale: any;
    rescale_factor: any;
    do_normalize: any;
    do_resize: any;
    do_thumbnail: any;
    size: any;
    do_center_crop: any;
    crop_size: any;
    do_convert_rgb: any;
    pad_size: any;
    do_pad: any;
    /**
     * Resize the image to make a thumbnail. The image is resized so that no dimension is larger than any
     * corresponding dimension of the specified size.
     * @param {RawImage} image The image to be resized.
     * @param {{height:number, width:number}} size The size `{"height": h, "width": w}` to resize the image to.
     * @param {string | 0 | 1 | 2 | 3 | 4 | 5} [resample=2] The resampling filter to use.
     * @returns {Promise<RawImage>} The resized image.
     */
    thumbnail(image: RawImage, size: {
        height: number;
        width: number;
    }, resample?: string | 0 | 1 | 2 | 3 | 4 | 5): Promise<RawImage>;
    /**
     * @typedef {object} PreprocessedImage
     * @property {HeightWidth} original_size The original size of the image.
     * @property {HeightWidth} reshaped_input_size The reshaped input size of the image.
     * @property {Tensor} pixel_values The pixel values of the preprocessed image.
     */
    /**
     * Preprocesses the given image.
     *
     * @param {RawImage} image The image to preprocess.
     * @returns {Promise<PreprocessedImage>} The preprocessed image.
     */
    preprocess(image: RawImage): Promise<{
        /**
         * The original size of the image.
         */
        original_size: HeightWidth;
        /**
         * The reshaped input size of the image.
         */
        reshaped_input_size: HeightWidth;
        /**
         * The pixel values of the preprocessed image.
         */
        pixel_values: Tensor;
    }>;
    /**
     * Calls the feature extraction process on an array of image
     * URLs, preprocesses each image, and concatenates the resulting
     * features into a single Tensor.
     * @param {any[]} images The URL(s) of the image(s) to extract features from.
     * @param {...any} args Additional arguments.
     * @returns {Promise<ImageFeatureExtractorResult>} An object containing the concatenated pixel values (and other metadata) of the preprocessed images.
     */
    _call(images: any[], ...args: any[]): Promise<ImageFeatureExtractorResult>;
}
export class ConvNextFeatureExtractor extends ImageFeatureExtractor {
}
export class ViTFeatureExtractor extends ImageFeatureExtractor {
}
export class MobileViTFeatureExtractor extends ImageFeatureExtractor {
}
export class DeiTFeatureExtractor extends ImageFeatureExtractor {
}
export class BeitFeatureExtractor extends ImageFeatureExtractor {
}
export class DonutFeatureExtractor extends ImageFeatureExtractor {
}
/**
 * @typedef {object} DetrFeatureExtractorResultProps
 * @property {Tensor} pixel_mask
 * @typedef {ImageFeatureExtractorResult & DetrFeatureExtractorResultProps} DetrFeatureExtractorResult
 */
/**
 * Detr Feature Extractor.
 *
 * @extends ImageFeatureExtractor
 */
export class DetrFeatureExtractor extends ImageFeatureExtractor {
    /**
     * Calls the feature extraction process on an array of image URLs, preprocesses
     * each image, and concatenates the resulting features into a single Tensor.
     * @param {any[]} urls The URL(s) of the image(s) to extract features from.
     * @returns {Promise<DetrFeatureExtractorResult>} An object containing the concatenated pixel values of the preprocessed images.
     */
    _call(urls: any[]): Promise<DetrFeatureExtractorResult>;
    /**
     * Post-processes the outputs of the model (for object detection).
     * @param {Object} outputs The outputs of the model that must be post-processed
     * @param {Tensor} outputs.logits The logits
     * @param {Tensor} outputs.pred_boxes The predicted boxes.
     * @return {Object[]} An array of objects containing the post-processed outputs.
     */
    post_process_object_detection(outputs: {
        logits: Tensor;
        pred_boxes: Tensor;
    }, threshold?: number, target_sizes?: any): any[];
    /**
     * Binarize the given masks using `object_mask_threshold`, it returns the associated values of `masks`, `scores` and `labels`.
     * @param {Tensor} class_logits The class logits.
     * @param {Tensor} mask_logits The mask logits.
     * @param {number} object_mask_threshold A number between 0 and 1 used to binarize the masks.
     * @param {number} num_labels The number of labels.
     * @returns {[Tensor[], number[], number[]]} The binarized masks, the scores, and the labels.
     */
    remove_low_and_no_objects(class_logits: Tensor, mask_logits: Tensor, object_mask_threshold: number, num_labels: number): [Tensor[], number[], number[]];
    /**
     * Checks whether the segment is valid or not.
     * @param {Int32Array} mask_labels Labels for each pixel in the mask.
     * @param {Tensor[]} mask_probs Probabilities for each pixel in the masks.
     * @param {number} k The class id of the segment.
     * @param {number} mask_threshold The mask threshold.
     * @param {number} overlap_mask_area_threshold The overlap mask area threshold.
     * @returns {[boolean, number[]]} Whether the segment is valid or not, and the indices of the valid labels.
     */
    check_segment_validity(mask_labels: Int32Array, mask_probs: Tensor[], k: number, mask_threshold?: number, overlap_mask_area_threshold?: number): [boolean, number[]];
    /**
     * Computes the segments.
     * @param {Tensor[]} mask_probs The mask probabilities.
     * @param {number[]} pred_scores The predicted scores.
     * @param {number[]} pred_labels The predicted labels.
     * @param {number} mask_threshold The mask threshold.
     * @param {number} overlap_mask_area_threshold The overlap mask area threshold.
     * @param {Set<number>} label_ids_to_fuse The label ids to fuse.
     * @param {number[]} target_size The target size of the image.
     * @returns {[Tensor, Array<{id: number, label_id: number, score: number}>]} The computed segments.
     */
    compute_segments(mask_probs: Tensor[], pred_scores: number[], pred_labels: number[], mask_threshold: number, overlap_mask_area_threshold: number, label_ids_to_fuse?: Set<number>, target_size?: number[]): [Tensor, Array<{
        id: number;
        label_id: number;
        score: number;
    }>];
    /**
     * Post-process the model output to generate the final panoptic segmentation.
     * @param {*} outputs The model output to post process
     * @param {number} [threshold=0.5] The probability score threshold to keep predicted instance masks.
     * @param {number} [mask_threshold=0.5] Threshold to use when turning the predicted masks into binary values.
     * @param {number} [overlap_mask_area_threshold=0.8] The overlap mask area threshold to merge or discard small disconnected parts within each binary instance mask.
     * @param {Set<number>} [label_ids_to_fuse=null] The labels in this state will have all their instances be fused together.
     * @param {number[][]} [target_sizes=null] The target sizes to resize the masks to.
     * @returns {Array<{ segmentation: Tensor, segments_info: Array<{id: number, label_id: number, score: number}>}>}
     */
    post_process_panoptic_segmentation(outputs: any, threshold?: number, mask_threshold?: number, overlap_mask_area_threshold?: number, label_ids_to_fuse?: Set<number>, target_sizes?: number[][]): Array<{
        segmentation: Tensor;
        segments_info: Array<{
            id: number;
            label_id: number;
            score: number;
        }>;
    }>;
    post_process_instance_segmentation(): void;
}
export class YolosFeatureExtractor extends ImageFeatureExtractor {
    /**
     * Post-processes the outputs of the model (for object detection).
     * @param {Object} outputs The outputs of the model that must be post-processed
     * @param {Tensor} outputs.logits The logits
     * @param {Tensor} outputs.pred_boxes The predicted boxes.
     * @return {Object[]} An array of objects containing the post-processed outputs.
     */
    post_process_object_detection(outputs: {
        logits: Tensor;
        pred_boxes: Tensor;
    }, threshold?: number, target_sizes?: any): any[];
}
/**
 * @typedef {object} SamImageProcessorResult
 * @property {Tensor} pixel_values
 * @property {HeightWidth[]} original_sizes
 * @property {HeightWidth[]} reshaped_input_sizes
 * @property {Tensor} input_points
 */
export class SamImageProcessor extends ImageFeatureExtractor {
    /**
     * @param {any[]} images The URL(s) of the image(s) to extract features from.
     * @param {*} input_points A 3D or 4D array, representing the input points provided by the user.
     * - 3D: `[point_batch_size, nb_points_per_image, 2]`. In this case, `batch_size` is assumed to be 1.
     * - 4D: `[batch_size, point_batch_size, nb_points_per_image, 2]`.
     * @returns {Promise<SamImageProcessorResult>}
     */
    _call(images: any[], input_points: any): Promise<SamImageProcessorResult>;
    /**
     * Remove padding and upscale masks to the original image size.
     * @param {Tensor} masks Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
     * @param {number[][]} original_sizes The original sizes of each image before it was resized to the model's expected input shape, in (height, width) format.
     * @param {number[][]} reshaped_input_sizes The size of each image as it is fed to the model, in (height, width) format. Used to remove padding.
     * @param {Object} options Optional parameters for post-processing.
     * @param {number} [options.mask_threshold] The threshold to use for binarizing the masks.
     * @param {boolean} [options.binarize] Whether to binarize the masks.
     * @param {Object} [options.pad_size] The target size the images were padded to before being passed to the model. If `null`, the target size is assumed to be the processor's `pad_size`.
     * @param {number} [options.pad_size.height] The height the images were padded to.
     * @param {number} [options.pad_size.width] The width the images were padded to.
     * @returns {Tensor[]} Batched masks in batch_size, num_channels, height, width) format, where (height, width) is given by original_size.
     */
    post_process_masks(masks: Tensor, original_sizes: number[][], reshaped_input_sizes: number[][], { mask_threshold, binarize, pad_size, }?: {
        mask_threshold?: number;
        binarize?: boolean;
        pad_size?: {
            height?: number;
            width?: number;
        };
    }): Tensor[];
}
export class WhisperFeatureExtractor extends FeatureExtractor {
    constructor(config: any);
    /**
     * Calculates the index offset for a given index and window size.
     * @param {number} i The index.
     * @param {number} w The window size.
     * @returns {number} The index offset.
     */
    calcOffset(i: number, w: number): number;
    /**
     * Pads an array with a reflected version of itself on both ends.
     * @param {Float32Array} array The array to pad.
     * @param {number} left The amount of padding to add to the left.
     * @param {number} right The amount of padding to add to the right.
     * @returns {Float32Array} The padded array.
     */
    padReflect(array: Float32Array, left: number, right: number): Float32Array;
    /**
     * Calculates the complex Short-Time Fourier Transform (STFT) of the given framed signal.
     *
     * @param {number[][]} frames A 2D array representing the signal frames.
     * @param {number[]} window A 1D array representing the window to be applied to the frames.
     * @returns {Object} An object with the following properties:
     * - data: A 1D array representing the complex STFT of the signal.
     * - dims: An array representing the dimensions of the STFT data, i.e. [num_frames, num_fft_bins].
     */
    stft(frames: number[][], window: number[]): any;
    /**
     * Creates an array of frames from a given waveform.
     *
     * @param {Float32Array} waveform The waveform to create frames from.
     * @param {boolean} [center=true] Whether to center the frames on their corresponding positions in the waveform. Defaults to true.
     * @returns {Array} An array of frames.
     */
    fram_wave(waveform: Float32Array, center?: boolean): any[];
    /**
     * Generates a Hanning window of length M.
     *
     * @param {number} M The length of the Hanning window to generate.
     * @returns {*} The generated Hanning window.
     */
    hanning(M: number): any;
    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {{data: Float32Array, dims: number[]}} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    _extract_fbank_features(waveform: Float32Array | Float64Array): {
        data: Float32Array;
        dims: number[];
    };
    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    _call(audio: Float32Array | Float64Array): Promise<{
        input_features: Tensor;
    }>;
}
export class Wav2Vec2FeatureExtractor extends FeatureExtractor {
    /**
     * @param {Float32Array} input_values
     * @returns {Float32Array}
     */
    _zero_mean_unit_var_norm(input_values: Float32Array): Float32Array;
    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_values: Tensor; attention_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention mask as Tensors.
     */
    _call(audio: Float32Array | Float64Array): Promise<{
        input_values: Tensor;
        attention_mask: Tensor;
    }>;
}
declare const Processor_base: new () => {
    (...args: any[]): any;
    _call(...args: any[]): any;
};
/**
 * Represents a Processor that extracts features from an input.
 * @extends Callable
 */
export class Processor extends Processor_base {
    /**
     * Creates a new Processor with the given feature extractor.
     * @param {FeatureExtractor} feature_extractor The function used to extract features from the input.
     */
    constructor(feature_extractor: FeatureExtractor);
    feature_extractor: FeatureExtractor;
    /**
     * Calls the feature_extractor function with the given input.
     * @param {any} input The input to extract features from.
     * @param {...any} args Additional arguments.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    _call(input: any, ...args: any[]): Promise<any>;
}
export class SamProcessor extends Processor {
    /**
     * @param {*} images
     * @param {*} input_points
     * @returns {Promise<any>}
     */
    _call(images: any, input_points: any): Promise<any>;
    /**
     * @borrows SamImageProcessor#post_process_masks as post_process_masks
     */
    post_process_masks(...args: any[]): any;
}
/**
 * Represents a WhisperProcessor that extracts features from an audio input.
 * @extends Processor
 */
export class WhisperProcessor extends Processor {
    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    _call(audio: any): Promise<any>;
}
export class Wav2Vec2ProcessorWithLM extends Processor {
    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    _call(audio: any): Promise<any>;
}
/**
 * Helper class which is used to instantiate pretrained processors with the `from_pretrained` function.
 * The chosen processor class is determined by the type specified in the processor config.
 *
 * **Example:** Load a processor using `from_pretrained`.
 * ```javascript
 * let processor = await AutoProcessor.from_pretrained('openai/whisper-tiny.en');
 * ```
 *
 * **Example:** Run an image through a processor.
 * ```javascript
 * let processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * let image_inputs = await processor(image);
 * // {
 * //   "pixel_values": {
 * //     "dims": [ 1, 3, 224, 224 ],
 * //     "type": "float32",
 * //     "data": Float32Array [ -1.558687686920166, -1.558687686920166, -1.5440893173217773, ... ],
 * //     "size": 150528
 * //   },
 * //   "original_sizes": [
 * //     [ 533, 800 ]
 * //   ],
 * //   "reshaped_input_sizes": [
 * //     [ 224, 224 ]
 * //   ]
 * // }
 * ```
 */
export class AutoProcessor {
    static FEATURE_EXTRACTOR_CLASS_MAPPING: {
        WhisperFeatureExtractor: typeof WhisperFeatureExtractor;
        ViTFeatureExtractor: typeof ViTFeatureExtractor;
        MobileViTFeatureExtractor: typeof MobileViTFeatureExtractor;
        ConvNextFeatureExtractor: typeof ConvNextFeatureExtractor;
        BeitFeatureExtractor: typeof BeitFeatureExtractor;
        DeiTFeatureExtractor: typeof DeiTFeatureExtractor;
        DetrFeatureExtractor: typeof DetrFeatureExtractor;
        YolosFeatureExtractor: typeof YolosFeatureExtractor;
        DonutFeatureExtractor: typeof DonutFeatureExtractor;
        SamImageProcessor: typeof SamImageProcessor;
        Wav2Vec2FeatureExtractor: typeof Wav2Vec2FeatureExtractor;
    };
    static PROCESSOR_CLASS_MAPPING: {
        WhisperProcessor: typeof WhisperProcessor;
        Wav2Vec2ProcessorWithLM: typeof Wav2Vec2ProcessorWithLM;
        SamProcessor: typeof SamProcessor;
    };
    /**
     * Instantiate one of the processor classes of the library from a pretrained model.
     *
     * The processor class to instantiate is selected based on the `feature_extractor_type` property of the config object
     * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     *
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
     * @param {import('./utils/hub.js').PretrainedOptions} options Additional options for loading the processor.
     *
     * @returns {Promise<Processor>} A new instance of the Processor class.
     */
    static from_pretrained(pretrained_model_name_or_path: string, { progress_callback, config, cache_dir, local_files_only, revision, }?: import('./utils/hub.js').PretrainedOptions): Promise<Processor>;
}
/**
 * Named tuple to indicate the order we are using is (height x width), even though
 * the Graphics’ industry standard is (width x height).
 */
export type HeightWidth = [height: number, width: number];
export type ImageFeatureExtractorResult = {
    /**
     * The pixel values of the batched preprocessed images.
     */
    pixel_values: Tensor;
    /**
     * Array of two-dimensional tuples like [[480, 640]].
     */
    original_sizes: HeightWidth[];
    /**
     * Array of two-dimensional tuples like [[1000, 1330]].
     */
    reshaped_input_sizes: HeightWidth[];
};
export type DetrFeatureExtractorResultProps = {
    pixel_mask: Tensor;
};
export type DetrFeatureExtractorResult = ImageFeatureExtractorResult & DetrFeatureExtractorResultProps;
export type SamImageProcessorResult = {
    pixel_values: Tensor;
    original_sizes: HeightWidth[];
    reshaped_input_sizes: HeightWidth[];
    input_points: Tensor;
};
import { RawImage } from './utils/image.js';
import { Tensor } from './utils/tensor.js';
export {};
//# sourceMappingURL=processors.d.ts.map