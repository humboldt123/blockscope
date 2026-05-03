package com.blockscope.recording;

import com.blockscope.upload.ChunkedVideoUploader;
import org.jcodec.codecs.h264.H264Encoder;
import org.jcodec.codecs.h264.H264Utils;
import org.jcodec.codecs.h264.encode.RateControl;
import org.jcodec.codecs.h264.encode.H264FixedRateControl;
import org.jcodec.common.VideoEncoder.EncodedFrame;
import org.jcodec.common.io.NIOUtils;
import org.jcodec.common.model.ColorSpace;
import org.jcodec.common.model.Picture;
import org.jcodec.common.model.Size;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Encodes video frames to H.264 and wraps in MPEG-TS packets for streaming.
 * Writes encoded data directly to ChunkedVideoUploader during recording.
 */
public class StreamingTSEncoder {
    private final ChunkedVideoUploader uploader;
    private final H264Encoder encoder;
    private final int width;
    private final int height;
    private final int fps;

    private int frameNumber = 0;
    private long pts = 0; // Presentation timestamp
    private static final int PTS_INCREMENT = 3003; // For 30fps (90000/30 = 3000, add small buffer)

    // MPEG-TS packet size
    private static final int TS_PACKET_SIZE = 188;
    private static final int TS_PAYLOAD_SIZE = 184; // 188 - 4 byte header

    // PIDs (Packet IDs)
    private static final int PAT_PID = 0x0000;
    private static final int PMT_PID = 0x1000;
    private static final int VIDEO_PID = 0x0100;

    // Continuity counters (4-bit, wraps at 16)
    private int patContinuity = 0;
    private int pmtContinuity = 0;
    private int videoContinuity = 0;

    private boolean headersSent = false;

    public StreamingTSEncoder(int width, int height, int fps, ChunkedVideoUploader uploader) {
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.uploader = uploader;

        // Initialize H264 encoder with fixed rate control (1024 Kbps)
        RateControl rc = new H264FixedRateControl(1024);
        this.encoder = new H264Encoder(rc);
    }

    /**
     * Start streaming - sends PAT and PMT tables
     */
    public void start() throws IOException {
        if (!headersSent) {
            sendPAT();
            sendPMT();
            headersSent = true;
        }
    }

    /**
     * Encode and stream a video frame
     */
    public void encodeFrame(BufferedImage image) throws IOException {
        // Convert BufferedImage to Picture (RGB - encoder will convert internally)
        Picture pic = org.jcodec.scale.AWTUtil.fromBufferedImage(image, org.jcodec.common.model.ColorSpace.RGB);

        // Encode to H.264
        ByteBuffer encoded = ByteBuffer.allocate(width * height * 3);
        EncodedFrame frame = encoder.encodeFrame(pic, encoded);

        if (frame != null && frame.getData() != null) {
            ByteBuffer h264Data = frame.getData();
            h264Data.rewind();

            // Wrap H.264 frame in PES packet, then in TS packets
            byte[] pesPacket = createPESPacket(h264Data, frame.isKeyFrame());

            // Send TS packets
            sendTSPackets(pesPacket, VIDEO_PID, frame.isKeyFrame());

            frameNumber++;
            pts += PTS_INCREMENT;
        }
    }

    /**
     * Finish streaming
     */
    public void finish() throws IOException {
        uploader.finish();
    }

    /**
     * Convert BufferedImage to YUV420 Picture
     */
    private Picture convertToYUV420(BufferedImage img) {
        Picture pic = Picture.create(width, height, ColorSpace.YUV420);

        int[] rgb = img.getRGB(0, 0, width, height, null, 0, width);

        byte[] y = pic.getPlaneData(0);
        byte[] u = pic.getPlaneData(1);
        byte[] v = pic.getPlaneData(2);

        int uOffset = 0;
        int vOffset = 0;

        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                int pixel = rgb[j * width + i];
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;

                // RGB to YUV conversion (fixed operator precedence)
                int yVal = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
                int uVal = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                int vVal = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

                y[j * width + i] = (byte) Math.max(0, Math.min(255, yVal));

                // Subsample U and V (4:2:0)
                if ((j % 2 == 0) && (i % 2 == 0)) {
                    u[uOffset] = (byte) Math.max(0, Math.min(255, uVal));
                    v[vOffset] = (byte) Math.max(0, Math.min(255, vVal));
                    uOffset++;
                    vOffset++;
                }
            }
        }

        return pic;
    }

    /**
     * Create PES (Packetized Elementary Stream) packet for H.264 data
     */
    private byte[] createPESPacket(ByteBuffer h264Data, boolean isKeyFrame) {
        ByteArrayOutputStream pes = new ByteArrayOutputStream();

        try {
            // PES header
            pes.write(0x00); pes.write(0x00); pes.write(0x01); // Start code
            pes.write(0xE0); // Stream ID (video)

            int pesPayloadSize = h264Data.remaining() + 8; // +8 for PTS/DTS fields
            pes.write((pesPayloadSize >> 8) & 0xFF);
            pes.write(pesPayloadSize & 0xFF);

            // PES flags
            pes.write(0x80); // '10' marker bits, no scrambling, no priority, data aligned
            pes.write(0x80); // PTS present, no DTS, no other flags
            pes.write(0x05); // PES header length (5 bytes for PTS)

            // PTS (33-bit timestamp)
            long ptsValue = pts;
            pes.write(0x21 | (int)((ptsValue >> 29) & 0x0E)); // '0010' + top 3 bits + '1'
            pes.write((int)((ptsValue >> 22) & 0xFF));
            pes.write(0x01 | (int)((ptsValue >> 14) & 0xFE));
            pes.write((int)((ptsValue >> 7) & 0xFF));
            pes.write(0x01 | (int)((ptsValue << 1) & 0xFE));

            // H.264 data
            byte[] h264Bytes = new byte[h264Data.remaining()];
            h264Data.get(h264Bytes);
            pes.write(h264Bytes);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return pes.toByteArray();
    }

    /**
     * Send TS packets containing PES data
     */
    private void sendTSPackets(byte[] pesData, int pid, boolean isKeyFrame) throws IOException {
        int offset = 0;
        boolean firstPacket = true;

        while (offset < pesData.length) {
            byte[] tsPacket = new byte[TS_PACKET_SIZE];

            // TS header (4 bytes)
            tsPacket[0] = 0x47; // Sync byte

            // PID and flags
            int pidHigh = (pid >> 8) & 0x1F;
            if (firstPacket) {
                tsPacket[1] = (byte) (0x40 | pidHigh); // PUSI (Payload Unit Start Indicator) set
            } else {
                tsPacket[1] = (byte) pidHigh;
            }
            tsPacket[2] = (byte) (pid & 0xFF);

            // Continuity counter
            tsPacket[3] = (byte) (0x10 | (videoContinuity & 0x0F)); // No adaptation, payload present
            videoContinuity = (videoContinuity + 1) % 16;

            // Payload
            int bytesToCopy = Math.min(TS_PAYLOAD_SIZE, pesData.length - offset);
            System.arraycopy(pesData, offset, tsPacket, 4, bytesToCopy);

            // Padding with 0xFF if needed
            for (int i = 4 + bytesToCopy; i < TS_PACKET_SIZE; i++) {
                tsPacket[i] = (byte) 0xFF;
            }

            // Write to uploader
            uploader.write(tsPacket);

            offset += bytesToCopy;
            firstPacket = false;
        }
    }

    /**
     * Send PAT (Program Association Table)
     */
    private void sendPAT() throws IOException {
        byte[] pat = new byte[TS_PACKET_SIZE];

        // TS header
        pat[0] = 0x47; // Sync byte
        pat[1] = 0x40; // PUSI set, PID high bits
        pat[2] = 0x00; // PID = 0x0000 (PAT)
        pat[3] = (byte) (0x10 | (patContinuity & 0x0F));
        patContinuity = (patContinuity + 1) % 16;

        // Pointer field
        pat[4] = 0x00;

        // PAT data
        pat[5] = 0x00; // Table ID (PAT)
        pat[6] = (byte) 0xB0; // Section syntax indicator, private bit
        pat[7] = 0x0D; // Section length (13 bytes)

        // Transport stream ID
        pat[8] = 0x00; pat[9] = 0x01;

        // Version and flags
        pat[10] = (byte) 0xC1; // Version 0, current/next indicator
        pat[11] = 0x00; // Section number
        pat[12] = 0x00; // Last section number

        // Program 1 -> PMT PID
        pat[13] = 0x00; pat[14] = 0x01; // Program number 1
        pat[15] = (byte) ((PMT_PID >> 8) | 0xE0);
        pat[16] = (byte) (PMT_PID & 0xFF);

        // CRC32 (simplified - should calculate properly)
        pat[17] = 0x00; pat[18] = 0x00; pat[19] = 0x00; pat[20] = 0x00;

        // Padding
        for (int i = 21; i < TS_PACKET_SIZE; i++) {
            pat[i] = (byte) 0xFF;
        }

        uploader.write(pat);
    }

    /**
     * Send PMT (Program Map Table)
     */
    private void sendPMT() throws IOException {
        byte[] pmt = new byte[TS_PACKET_SIZE];

        // TS header
        pmt[0] = 0x47; // Sync byte
        pmt[1] = (byte) (0x40 | ((PMT_PID >> 8) & 0x1F)); // PUSI set
        pmt[2] = (byte) (PMT_PID & 0xFF);
        pmt[3] = (byte) (0x10 | (pmtContinuity & 0x0F));
        pmtContinuity = (pmtContinuity + 1) % 16;

        // Pointer field
        pmt[4] = 0x00;

        // PMT data
        pmt[5] = 0x02; // Table ID (PMT)
        pmt[6] = (byte) 0xB0; // Section syntax indicator
        pmt[7] = 0x12; // Section length (18 bytes)

        // Program number
        pmt[8] = 0x00; pmt[9] = 0x01;

        // Version and flags
        pmt[10] = (byte) 0xC1; // Version 0, current/next indicator
        pmt[11] = 0x00; // Section number
        pmt[12] = 0x00; // Last section number

        // PCR PID (use video PID)
        pmt[13] = (byte) ((VIDEO_PID >> 8) | 0xE0);
        pmt[14] = (byte) (VIDEO_PID & 0xFF);

        // Program info length (0 - no descriptors)
        pmt[15] = (byte) 0xF0; pmt[16] = 0x00;

        // Elementary stream: H.264 video
        pmt[17] = 0x1B; // Stream type: H.264
        pmt[18] = (byte) ((VIDEO_PID >> 8) | 0xE0);
        pmt[19] = (byte) (VIDEO_PID & 0xFF);
        pmt[20] = (byte) 0xF0; pmt[21] = 0x00; // ES info length (0)

        // CRC32 (simplified)
        pmt[22] = 0x00; pmt[23] = 0x00; pmt[24] = 0x00; pmt[25] = 0x00;

        // Padding
        for (int i = 26; i < TS_PACKET_SIZE; i++) {
            pmt[i] = (byte) 0xFF;
        }

        uploader.write(pmt);
    }
}
