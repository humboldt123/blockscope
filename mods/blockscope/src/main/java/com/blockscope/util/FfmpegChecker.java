package com.blockscope.util;

/**
 * Checks at startup whether ffmpeg is on PATH and functional.
 * Result is cached after the first call.
 */
public class FfmpegChecker {

    private static Boolean available = null;
    private static String  version   = null;

    private FfmpegChecker() {}

    /** Runs the check (once) and returns true if ffmpeg -version exits 0. */
    public static boolean isAvailable() {
        if (available == null) probe();
        return available;
    }

    /** Short version string from ffmpeg -version, or "not found" if unavailable. */
    public static String getVersionString() {
        if (available == null) probe();
        return version != null ? version : "not found";
    }

    /** Eagerly probe ffmpeg so the result is ready before any login handshake. */
    public static void probe() {
        try {
            Process proc = new ProcessBuilder("ffmpeg", "-version")
                .redirectErrorStream(true)
                .start();

            byte[] out = proc.getInputStream().readAllBytes();
            int exit = proc.waitFor();

            if (exit == 0) {
                available = true;
                // First line is e.g. "ffmpeg version 6.0 Copyright ..."
                String first = new String(out).split("\n")[0].trim();
                version = first.length() > 80 ? first.substring(0, 80) : first;
            } else {
                available = false;
            }
        } catch (Exception e) {
            available = false;
        }
        System.out.println("[Blockscope] ffmpeg check: " + (available ? "OK — " + version : "NOT FOUND"));
    }
}
