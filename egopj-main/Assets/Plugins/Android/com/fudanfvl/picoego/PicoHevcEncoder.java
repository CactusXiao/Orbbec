package com.fudanfvl.picoego;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.os.Build;

import org.json.JSONObject;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.HashMap;

public class PicoHevcEncoder {
    private static final String MIME_TYPE = "video/hevc";
    private static final int COLOR_QCOM_YUV420_SEMIPLANAR = 0x7FA30C00;
    private static final long EOS_INPUT_DEQUEUE_TIMEOUT_US = 100000L;
    private static final long EOS_OUTPUT_DEQUEUE_TIMEOUT_US = 100000L;
    private static final long EOS_DRAIN_TIMEOUT_NS = 5000000000L;
    private static final int EOS_INPUT_MAX_ATTEMPTS = 10;
    private static final int EOS_MAX_TRY_AGAIN_COUNT = 50;
    private static final int EOS_MAX_DRAIN_ITERATIONS = 10000;
    private static final int MAX_PENDING_FRAME_META_COUNT = 256;

    private MediaCodec codec;
    private MediaCodec.BufferInfo bufferInfo;
    private final ArrayDeque<EncodedSample> pendingSamples = new ArrayDeque<>();
    private final HashMap<Long, FrameMeta> frameMetasByPts = new HashMap<>();

    private byte[] conversionBuffer;
    private byte[] directChromaBuffer;
    private int width;
    private int height;
    private int bitRate;
    private int frameRate;
    private int iFrameIntervalSeconds;
    private int selectedColorFormat;
    private String encoderName = "";
    private String lastError = "";
    private boolean started;
    private boolean sawOutputFormat;
    private long lastTotalEncodeNs;
    private long lastDequeueInputNs;
    private long lastColorConvertNs;
    private long lastInputPutNs;
    private long lastQueueInputNs;
    private long lastDrainOutputNs;
    private long submittedInputFrameCount;
    private long matchedOutputFrameCount;
    private long unknownOutputPtsCount;
    private long duplicateInputPtsCount;
    private long nonMonotonicInputPtsCount;
    private long partialOutputSampleCount;
    private int frameMetaHighWaterMark;
    private int frameMetaCountAfterEos;
    private long lastSubmittedPresentationTimeUs;
    private boolean hasSubmittedPresentationTimeUs;
    private boolean sawEndOfStream;

    private static class FrameMeta {
        int frameIndex;
        long refTimestampUs;

        FrameMeta(int frameIndex, long refTimestampUs) {
            this.frameIndex = frameIndex;
            this.refTimestampUs = refTimestampUs;
        }
    }

    private static class EncodedSample {
        String headerJson;
        byte[] data;

        EncodedSample(String headerJson, byte[] data) {
            this.headerJson = headerJson;
            this.data = data;
        }
    }

    public boolean start(int width, int height, int bitRate, int frameRate, int iFrameIntervalSeconds) {
        stop();
        lastError = "";

        if ((width & 1) != 0 || (height & 1) != 0 || width <= 0 || height <= 0) {
            lastError = "HEVC input width/height must be positive even numbers.";
            return false;
        }

        this.width = width;
        this.height = height;
        this.bitRate = bitRate;
        this.frameRate = frameRate;
        this.iFrameIntervalSeconds = iFrameIntervalSeconds;

        try {
            EncoderSelection selection = selectEncoder();
            if (selection == null) {
                lastError = "No HEVC encoder with supported byte-buffer YUV420 color format was found.";
                return false;
            }

            encoderName = selection.name;
            selectedColorFormat = selection.colorFormat;
            codec = MediaCodec.createByCodecName(encoderName);

            MediaFormat format = MediaFormat.createVideoFormat(MIME_TYPE, width, height);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, selectedColorFormat);
            format.setInteger(MediaFormat.KEY_BIT_RATE, bitRate);
            format.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate);
            format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, iFrameIntervalSeconds);
            if (Build.VERSION.SDK_INT >= 23) {
                format.setInteger(MediaFormat.KEY_PRIORITY, 0);
            }
            if (Build.VERSION.SDK_INT >= 26) {
                format.setInteger(MediaFormat.KEY_LATENCY, 0);
            }

            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE);
            codec.start();
            bufferInfo = new MediaCodec.BufferInfo();
            conversionBuffer = new byte[width * height * 3 / 2];
            directChromaBuffer = new byte[width * height / 2];
            pendingSamples.clear();
            frameMetasByPts.clear();
            sawOutputFormat = false;
            resetLastTimingStats();
            resetFrameTrackingStats();
            started = true;
            return true;
        } catch (Exception ex) {
            lastError = ex.getClass().getSimpleName() + ": " + ex.getMessage();
            stop();
            return false;
        }
    }

    public int encodeFrame(byte[] nv21, long presentationTimeUs, int frameIndex, long refTimestampUs) {
        if (!started || codec == null) {
            lastError = "Encoder is not started.";
            return -1;
        }
        if (nv21 == null || nv21.length < width * height * 3 / 2) {
            lastError = "Invalid NV21 frame size.";
            return -2;
        }
        lastError = "";
        int ptsValidationResult = validateInputPresentationTimeUs(presentationTimeUs);
        if (ptsValidationResult != 0) {
            return ptsValidationResult;
        }

        try {
            long totalStartNs = System.nanoTime();
            long stageStartNs = totalStartNs;
            int inputIndex = codec.dequeueInputBuffer(10000);
            lastDequeueInputNs = System.nanoTime() - stageStartNs;
            if (inputIndex < 0) {
                lastError = "No HEVC input buffer available within timeout.";
                return -3;
            }

            ByteBuffer inputBuffer = codec.getInputBuffer(inputIndex);
            if (inputBuffer == null) {
                lastError = "HEVC input buffer is null.";
                return -4;
            }

            inputBuffer.clear();
            stageStartNs = System.nanoTime();
            convertNv21ToSelectedFormat(nv21, conversionBuffer);
            lastColorConvertNs = System.nanoTime() - stageStartNs;
            stageStartNs = System.nanoTime();
            inputBuffer.put(conversionBuffer, 0, conversionBuffer.length);
            lastInputPutNs = System.nanoTime() - stageStartNs;
            stageStartNs = System.nanoTime();
            codec.queueInputBuffer(inputIndex, 0, conversionBuffer.length, presentationTimeUs, 0);
            lastQueueInputNs = System.nanoTime() - stageStartNs;
            registerSubmittedFrame(presentationTimeUs, frameIndex, refTimestampUs);
            stageStartNs = System.nanoTime();
            drainOutput(false);
            lastDrainOutputNs = System.nanoTime() - stageStartNs;
            lastTotalEncodeNs = System.nanoTime() - totalStartNs;
            return pendingSamples.size();
        } catch (Exception ex) {
            lastError = ex.getClass().getSimpleName() + ": " + ex.getMessage();
            return -5;
        }
    }

    public int encodeFrameDirect(ByteBuffer nv21Buffer, int size, long presentationTimeUs, int frameIndex, long refTimestampUs) {
        if (!started || codec == null) {
            lastError = "Encoder is not started.";
            return -1;
        }
        int frameSize = width * height * 3 / 2;
        if (nv21Buffer == null || size < frameSize || nv21Buffer.capacity() < frameSize) {
            lastError = "Invalid direct NV21 frame buffer.";
            return -2;
        }
        lastError = "";
        int ptsValidationResult = validateInputPresentationTimeUs(presentationTimeUs);
        if (ptsValidationResult != 0) {
            return ptsValidationResult;
        }

        try {
            long totalStartNs = System.nanoTime();
            long stageStartNs = totalStartNs;
            int inputIndex = codec.dequeueInputBuffer(10000);
            lastDequeueInputNs = System.nanoTime() - stageStartNs;
            if (inputIndex < 0) {
                lastError = "No HEVC input buffer available within timeout.";
                return -3;
            }

            ByteBuffer inputBuffer = codec.getInputBuffer(inputIndex);
            if (inputBuffer == null) {
                lastError = "HEVC input buffer is null.";
                return -4;
            }

            inputBuffer.clear();
            stageStartNs = System.nanoTime();
            convertNv21DirectToSelectedFormat(nv21Buffer, conversionBuffer);
            lastColorConvertNs = System.nanoTime() - stageStartNs;
            stageStartNs = System.nanoTime();
            inputBuffer.put(conversionBuffer, 0, conversionBuffer.length);
            lastInputPutNs = System.nanoTime() - stageStartNs;
            stageStartNs = System.nanoTime();
            codec.queueInputBuffer(inputIndex, 0, conversionBuffer.length, presentationTimeUs, 0);
            lastQueueInputNs = System.nanoTime() - stageStartNs;
            registerSubmittedFrame(presentationTimeUs, frameIndex, refTimestampUs);
            stageStartNs = System.nanoTime();
            drainOutput(false);
            lastDrainOutputNs = System.nanoTime() - stageStartNs;
            lastTotalEncodeNs = System.nanoTime() - totalStartNs;
            return pendingSamples.size();
        } catch (Exception ex) {
            lastError = ex.getClass().getSimpleName() + ": " + ex.getMessage();
            return -5;
        }
    }

    public int finish(long presentationTimeUs) {
        if (!started || codec == null) {
            return pendingSamples.size();
        }
        lastError = "";

        try {
            int inputIndex = -1;
            for (int attempt = 0; attempt < EOS_INPUT_MAX_ATTEMPTS && inputIndex < 0; attempt++) {
                inputIndex = codec.dequeueInputBuffer(EOS_INPUT_DEQUEUE_TIMEOUT_US);
                if (inputIndex < 0) {
                    // Releasing pending output can unblock an encoder whose input side is back-pressured.
                    drainOutput(false);
                }
            }
            if (inputIndex >= 0) {
                long eosPresentationTimeUs = hasSubmittedPresentationTimeUs && presentationTimeUs <= lastSubmittedPresentationTimeUs
                        ? lastSubmittedPresentationTimeUs + 1L
                        : presentationTimeUs;
                codec.queueInputBuffer(inputIndex, 0, 0, eosPresentationTimeUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                drainOutput(true);
            } else {
                lastError = "No HEVC input buffer available for EOS after " +
                        EOS_INPUT_MAX_ATTEMPTS + " attempts.";
            }
        } catch (Exception ex) {
            lastError = ex.getClass().getSimpleName() + ": " + ex.getMessage();
        } finally {
            frameMetaCountAfterEos = frameMetasByPts.size();
        }

        return pendingSamples.size();
    }

    public void stop() {
        started = false;
        pendingSamples.clear();
        frameMetasByPts.clear();
        sawOutputFormat = false;
        resetLastTimingStats();
        resetFrameTrackingStats();

        if (codec != null) {
            try {
                codec.stop();
            } catch (Exception ignored) {
            }
            try {
                codec.release();
            } catch (Exception ignored) {
            }
            codec = null;
        }
    }

    public int getPendingSampleCount() {
        return pendingSamples.size();
    }

    public String peekSampleHeaderJson() {
        EncodedSample sample = pendingSamples.peek();
        return sample == null ? "" : sample.headerJson;
    }

    public byte[] popSampleData() {
        EncodedSample sample = pendingSamples.poll();
        return sample == null ? new byte[0] : sample.data;
    }

    public String getLastError() {
        return lastError == null ? "" : lastError;
    }

    public String getConfigJson() {
        try {
            JSONObject json = new JSONObject();
            json.put("mime_type", MIME_TYPE);
            json.put("encoder_name", encoderName);
            json.put("width", width);
            json.put("height", height);
            json.put("bit_rate", bitRate);
            json.put("frame_rate", frameRate);
            json.put("i_frame_interval_seconds", iFrameIntervalSeconds);
            json.put("selected_color_format", selectedColorFormat);
            json.put("selected_color_format_name", colorFormatName(selectedColorFormat));
            json.put("supports_direct_byte_buffer_input", true);
            json.put("max_pending_frame_meta_count", MAX_PENDING_FRAME_META_COUNT);
            return json.toString();
        } catch (Exception ex) {
            return "{\"error\":\"" + escapeJson(ex.getMessage()) + "\"}";
        }
    }

    public long getLastTotalEncodeNs() {
        return lastTotalEncodeNs;
    }

    public long getLastDequeueInputNs() {
        return lastDequeueInputNs;
    }

    public long getLastColorConvertNs() {
        return lastColorConvertNs;
    }

    public long getLastInputPutNs() {
        return lastInputPutNs;
    }

    public long getLastQueueInputNs() {
        return lastQueueInputNs;
    }

    public long getLastDrainOutputNs() {
        return lastDrainOutputNs;
    }

    public long getSubmittedInputFrameCount() {
        return submittedInputFrameCount;
    }

    public long getMatchedOutputFrameCount() {
        return matchedOutputFrameCount;
    }

    public long getUnknownOutputPtsCount() {
        return unknownOutputPtsCount;
    }

    public long getDuplicateInputPtsCount() {
        return duplicateInputPtsCount;
    }

    public int getFrameMetaHighWaterMark() {
        return frameMetaHighWaterMark;
    }

    public int getFrameMetaCountAfterEos() {
        return frameMetaCountAfterEos;
    }

    public long getNonMonotonicInputPtsCount() {
        return nonMonotonicInputPtsCount;
    }

    public long getPartialOutputSampleCount() {
        return partialOutputSampleCount;
    }

    public int getCurrentFrameMetaCount() {
        return frameMetasByPts.size();
    }

    public boolean getSawEndOfStream() {
        return sawEndOfStream;
    }

    private void resetLastTimingStats() {
        lastTotalEncodeNs = 0L;
        lastDequeueInputNs = 0L;
        lastColorConvertNs = 0L;
        lastInputPutNs = 0L;
        lastQueueInputNs = 0L;
        lastDrainOutputNs = 0L;
    }

    private void resetFrameTrackingStats() {
        submittedInputFrameCount = 0L;
        matchedOutputFrameCount = 0L;
        unknownOutputPtsCount = 0L;
        duplicateInputPtsCount = 0L;
        nonMonotonicInputPtsCount = 0L;
        partialOutputSampleCount = 0L;
        frameMetaHighWaterMark = 0;
        frameMetaCountAfterEos = 0;
        lastSubmittedPresentationTimeUs = 0L;
        hasSubmittedPresentationTimeUs = false;
        sawEndOfStream = false;
    }

    private int validateInputPresentationTimeUs(long presentationTimeUs) {
        if (frameMetasByPts.containsKey(presentationTimeUs) ||
                (hasSubmittedPresentationTimeUs && presentationTimeUs == lastSubmittedPresentationTimeUs)) {
            duplicateInputPtsCount++;
            lastError = "Duplicate HEVC input presentation timestamp: " + presentationTimeUs;
            return -6;
        }
        if (hasSubmittedPresentationTimeUs && presentationTimeUs < lastSubmittedPresentationTimeUs) {
            nonMonotonicInputPtsCount++;
            lastError = "Non-monotonic HEVC input presentation timestamp: " + presentationTimeUs +
                    " < " + lastSubmittedPresentationTimeUs;
            return -7;
        }
        if (frameMetasByPts.size() >= MAX_PENDING_FRAME_META_COUNT) {
            lastError = "HEVC pending frame metadata reached the safety limit of " +
                    MAX_PENDING_FRAME_META_COUNT + ".";
            return -8;
        }
        return 0;
    }

    private void registerSubmittedFrame(long presentationTimeUs, int frameIndex, long refTimestampUs) {
        frameMetasByPts.put(presentationTimeUs, new FrameMeta(frameIndex, refTimestampUs));
        lastSubmittedPresentationTimeUs = presentationTimeUs;
        hasSubmittedPresentationTimeUs = true;
        submittedInputFrameCount++;
        frameMetaHighWaterMark = Math.max(frameMetaHighWaterMark, frameMetasByPts.size());
    }

    private void drainOutput(boolean endOfStream) throws Exception {
        long drainStartNs = endOfStream ? System.nanoTime() : 0L;
        int tryAgainCount = 0;
        int drainIterations = 0;
        while (true) {
            if (endOfStream && (System.nanoTime() - drainStartNs >= EOS_DRAIN_TIMEOUT_NS ||
                    drainIterations >= EOS_MAX_DRAIN_ITERATIONS)) {
                throw new IllegalStateException("Timed out draining HEVC output after EOS. pending_frame_meta=" +
                        frameMetasByPts.size());
            }

            drainIterations++;
            int outputIndex = codec.dequeueOutputBuffer(
                    bufferInfo,
                    endOfStream ? EOS_OUTPUT_DEQUEUE_TIMEOUT_US : 0L);
            if (outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER) {
                if (!endOfStream) {
                    break;
                }
                tryAgainCount++;
                if (tryAgainCount >= EOS_MAX_TRY_AGAIN_COUNT) {
                    throw new IllegalStateException("Timed out waiting for HEVC EOS output. pending_frame_meta=" +
                            frameMetasByPts.size());
                }
                continue;
            }
            if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                sawOutputFormat = true;
                queueCodecConfigFromFormat(codec.getOutputFormat());
                continue;
            }
            if (outputIndex < 0) {
                continue;
            }
            tryAgainCount = 0;

            boolean sampleMapped = true;
            boolean sawEos = (bufferInfo.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            try {
                ByteBuffer outputBuffer = codec.getOutputBuffer(outputIndex);
                if (outputBuffer != null && bufferInfo.size > 0) {
                    outputBuffer.position(bufferInfo.offset);
                    outputBuffer.limit(bufferInfo.offset + bufferInfo.size);
                    byte[] data = new byte[bufferInfo.size];
                    outputBuffer.get(data);
                    sampleMapped = queueSample(data, bufferInfo.presentationTimeUs, bufferInfo.flags);
                }
            } finally {
                codec.releaseOutputBuffer(outputIndex, false);
            }

            if (!sampleMapped) {
                throw new IllegalStateException(lastError);
            }
            if (sawEos) {
                sawEndOfStream = true;
                break;
            }
        }
    }

    private void queueCodecConfigFromFormat(MediaFormat outputFormat) {
        if (outputFormat == null) {
            return;
        }

        try {
            ByteBuffer csd0 = outputFormat.getByteBuffer("csd-0");
            if (csd0 == null) {
                return;
            }
            ByteBuffer duplicate = csd0.duplicate();
            duplicate.position(0);
            byte[] data = new byte[duplicate.remaining()];
            duplicate.get(data);
            queueSample(data, 0L, MediaCodec.BUFFER_FLAG_CODEC_CONFIG);
        } catch (Exception ignored) {
        }
    }

    private boolean queueSample(byte[] data, long presentationTimeUs, int flags) throws Exception {
        boolean isCodecConfig = (flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0;
        boolean isPartialFrame = (flags & MediaCodec.BUFFER_FLAG_PARTIAL_FRAME) != 0;
        FrameMeta meta = isCodecConfig ? null : frameMetasByPts.get(presentationTimeUs);
        if (!isCodecConfig && meta == null) {
            unknownOutputPtsCount++;
            lastError = "HEVC output has no matching input PTS: " + presentationTimeUs;
            return false;
        }

        int frameIndex = meta == null ? -1 : meta.frameIndex;
        long refTimestampUs = isCodecConfig ? 0L : meta.refTimestampUs;
        boolean isKeyFrame = (flags & MediaCodec.BUFFER_FLAG_KEY_FRAME) != 0;

        JSONObject header = new JSONObject();
        header.put("frame_index", frameIndex);
        header.put("ref_timestamp_us", refTimestampUs);
        header.put("presentation_time_us", presentationTimeUs);
        header.put("is_keyframe", isKeyFrame);
        header.put("is_codec_config", isCodecConfig);
        header.put("is_partial_frame", isPartialFrame);
        header.put("flags", flags);
        header.put("width", width);
        header.put("height", height);
        header.put("size", data.length);
        header.put("encoder_name", encoderName);
        header.put("saw_output_format", sawOutputFormat);
        pendingSamples.add(new EncodedSample(header.toString(), data));

        if (!isCodecConfig) {
            if (isPartialFrame) {
                partialOutputSampleCount++;
            } else if (meta != null) {
                frameMetasByPts.remove(presentationTimeUs);
                matchedOutputFrameCount++;
            }
        }
        return true;
    }

    private void convertNv21ToSelectedFormat(byte[] nv21, byte[] output) {
        if (isPlanarFormat(selectedColorFormat)) {
            convertNv21ToI420(nv21, output);
        } else {
            convertNv21ToNv12(nv21, output);
        }
    }

    private void convertNv21DirectToSelectedFormat(ByteBuffer nv21, byte[] output) {
        if (isPlanarFormat(selectedColorFormat)) {
            convertNv21DirectToI420(nv21, output);
        } else {
            convertNv21DirectToNv12(nv21, output);
        }
    }

    private void convertNv21ToNv12(byte[] nv21, byte[] output) {
        int ySize = width * height;
        System.arraycopy(nv21, 0, output, 0, ySize);
        for (int i = 0; i < ySize / 2; i += 2) {
            output[ySize + i] = nv21[ySize + i + 1];
            output[ySize + i + 1] = nv21[ySize + i];
        }
    }

    private void convertNv21ToI420(byte[] nv21, byte[] output) {
        int ySize = width * height;
        int chromaSize = ySize / 4;
        int uOffset = ySize;
        int vOffset = ySize + chromaSize;
        System.arraycopy(nv21, 0, output, 0, ySize);
        for (int i = 0; i < chromaSize; i++) {
            output[uOffset + i] = nv21[ySize + i * 2 + 1];
            output[vOffset + i] = nv21[ySize + i * 2];
        }
    }

    private void convertNv21DirectToNv12(ByteBuffer nv21, byte[] output) {
        int ySize = width * height;
        int chromaBytes = ySize / 2;
        ByteBuffer input = nv21.duplicate();
        input.position(0);
        input.limit(ySize);
        input.get(output, 0, ySize);
        input.limit(ySize + chromaBytes);
        input.position(ySize);
        input.get(directChromaBuffer, 0, chromaBytes);
        for (int i = 0; i < chromaBytes; i += 2) {
            output[ySize + i] = directChromaBuffer[i + 1];
            output[ySize + i + 1] = directChromaBuffer[i];
        }
    }

    private void convertNv21DirectToI420(ByteBuffer nv21, byte[] output) {
        int ySize = width * height;
        int chromaSize = ySize / 4;
        int chromaBytes = ySize / 2;
        int uOffset = ySize;
        int vOffset = ySize + chromaSize;
        ByteBuffer input = nv21.duplicate();
        input.position(0);
        input.limit(ySize);
        input.get(output, 0, ySize);
        input.limit(ySize + chromaBytes);
        input.position(ySize);
        input.get(directChromaBuffer, 0, chromaBytes);
        for (int i = 0; i < chromaSize; i++) {
            output[uOffset + i] = directChromaBuffer[i * 2 + 1];
            output[vOffset + i] = directChromaBuffer[i * 2];
        }
    }

    private boolean isPlanarFormat(int colorFormat) {
        return colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar;
    }

    private EncoderSelection selectEncoder() {
        MediaCodecList codecList = new MediaCodecList(MediaCodecList.ALL_CODECS);
        MediaCodecInfo[] infos = codecList.getCodecInfos();
        for (MediaCodecInfo info : infos) {
            if (!info.isEncoder()) {
                continue;
            }
            for (String type : info.getSupportedTypes()) {
                if (!MIME_TYPE.equalsIgnoreCase(type)) {
                    continue;
                }
                MediaCodecInfo.CodecCapabilities caps = info.getCapabilitiesForType(type);
                int colorFormat = chooseColorFormat(caps.colorFormats);
                if (colorFormat != 0) {
                    return new EncoderSelection(info.getName(), colorFormat);
                }
            }
        }
        return null;
    }

    private int chooseColorFormat(int[] colorFormats) {
        int[] preferred = new int[]{
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible,
                COLOR_QCOM_YUV420_SEMIPLANAR,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar
        };

        for (int wanted : preferred) {
            for (int actual : colorFormats) {
                if (actual == wanted) {
                    return actual;
                }
            }
        }
        return 0;
    }

    private String colorFormatName(int colorFormat) {
        if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420SemiPlanar) {
            return "COLOR_FormatYUV420SemiPlanar";
        }
        if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible) {
            return "COLOR_FormatYUV420Flexible";
        }
        if (colorFormat == MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar) {
            return "COLOR_FormatYUV420Planar";
        }
        if (colorFormat == COLOR_QCOM_YUV420_SEMIPLANAR) {
            return "COLOR_QCOM_FormatYUV420SemiPlanar";
        }
        return "0x" + Integer.toHexString(colorFormat);
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static class EncoderSelection {
        String name;
        int colorFormat;

        EncoderSelection(String name, int colorFormat) {
            this.name = name;
            this.colorFormat = colorFormat;
        }
    }
}
