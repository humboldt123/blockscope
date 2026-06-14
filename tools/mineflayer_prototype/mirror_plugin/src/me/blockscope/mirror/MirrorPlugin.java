package me.blockscope.mirror;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitTask;
import org.bukkit.util.Vector;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * Minimal teleport-mirror plugin for the Blockscope camera-rig prototype.
 *
 * <p>A single controller player drives a single camera player. Every server tick
 * the camera is teleported to the controller's exact {@link Location} (which carries
 * yaw + pitch), so the camera's look direction mirrors the controller for free.
 * The camera is made into a free-floating, non-colliding, gravity-less observer and
 * is hidden from / can't see the controller (and vice-versa) so neither appears in
 * the other's recording or view.
 *
 * <p>This is a deliberately stripped-down port of solaris-engine's CameraManager:
 * NO ProtocolLib (no block-break animation relay), NO 1.20.5+ Attribute.GENERIC_SCALE
 * sneak-scaling, and only ONE pair instead of a pair map. It targets Paper 1.19.4
 * (api-version 1.19).
 *
 * <p>Recording on the camera (our Blockscope Fabric client) is triggered by sending
 * it the {@code blockscope:session_start} plugin message — the exact packet
 * Lodestone sends in production. Payload: {@code byte modeLen, byte[] mode,
 * int durationSeconds}. With no baritone jar present the client treats this as a
 * record-only trigger.
 *
 * <p>Commands:
 * <pre>
 *   /mirror &lt;controller&gt; &lt;camera&gt;   pair the two players, prep the camera, start the follow task
 *   /mirror stop                          unpair and restore the camera
 *   /startcam [seconds]                   send session_start to the camera (begin recording)
 * </pre>
 */
public class MirrorPlugin extends JavaPlugin {

    /** Plugin message channel the Blockscope client listens on (see SessionProtocol.java). */
    public static final String SESSION_CHANNEL = "blockscope:session_start";

    private volatile Player controller;
    private volatile Player camera;
    private BukkitTask followTask;

    @Override
    public void onEnable() {
        // Register the outgoing plugin channel so we can push session_start to the camera.
        getServer().getMessenger().registerOutgoingPluginChannel(this, SESSION_CHANNEL);
        getLogger().info("MirrorPlugin enabled (api 1.19, no ProtocolLib).");
    }

    @Override
    public void onDisable() {
        stopFollowingTask();
        if (camera != null) restoreCamera(camera);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args) {
        switch (cmd.getName().toLowerCase()) {
            case "mirror":
                return handleMirror(sender, args);
            case "startcam":
                return handleStartCam(sender, args);
            default:
                return false;
        }
    }

    // ------------------------------------------------------------------ /mirror

    private boolean handleMirror(CommandSender sender, String[] args) {
        if (args.length == 1 && args[0].equalsIgnoreCase("stop")) {
            stopFollowingTask();
            if (camera != null) restoreCamera(camera);
            controller = null;
            camera = null;
            sender.sendMessage("[mirror] stopped and restored.");
            return true;
        }

        if (args.length != 2) {
            sender.sendMessage("[mirror] usage: /mirror <controller> <camera> | /mirror stop");
            return true;
        }

        Player c = Bukkit.getPlayerExact(args[0]);
        Player k = Bukkit.getPlayerExact(args[1]);
        if (c == null) { sender.sendMessage("[mirror] controller '" + args[0] + "' not online."); return true; }
        if (k == null) { sender.sendMessage("[mirror] camera '" + args[1] + "' not online."); return true; }
        if (c.equals(k)) { sender.sendMessage("[mirror] controller and camera must differ."); return true; }

        controller = c;
        camera = k;

        prepareCamera(k);

        // Mutual invisibility so neither shows up in the other's view / recording.
        c.hidePlayer(this, k);
        k.hidePlayer(this, c);

        // Snap the camera onto the controller immediately, then start the per-tick follow.
        k.teleport(c.getLocation().clone());
        startFollowingTask();

        sender.sendMessage("[mirror] " + c.getName() + " -> camera " + k.getName() + " (following). "
                + "Use /startcam to begin recording.");
        getLogger().info("Paired controller=" + c.getName() + " camera=" + k.getName());
        return true;
    }

    // ---------------------------------------------------------------- /startcam

    private boolean handleStartCam(CommandSender sender, String[] args) {
        if (camera == null || !camera.isOnline()) {
            sender.sendMessage("[startcam] no camera paired/online — run /mirror first.");
            return true;
        }
        int seconds = 600;
        if (args.length >= 1) {
            try { seconds = Integer.parseInt(args[0]); } catch (NumberFormatException ignored) {}
        }
        sendSessionStart(camera, "adventure", seconds);
        sender.sendMessage("[startcam] sent session_start (mode=adventure, " + seconds + "s) to " + camera.getName());
        getLogger().info("Sent session_start to camera " + camera.getName() + " duration=" + seconds + "s");
        return true;
    }

    /**
     * Sends the blockscope:session_start plugin message. Payload byte layout MUST match
     * SessionProtocol.java on the client:
     *   byte    modeLen
     *   byte[]  mode (UTF-8)
     *   int     durationSeconds   (big-endian, DataOutputStream default)
     */
    private void sendSessionStart(Player cam, String mode, int durationSeconds) {
        try {
            byte[] modeBytes = mode.getBytes(StandardCharsets.UTF_8);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            DataOutputStream out = new DataOutputStream(baos);
            out.writeByte(modeBytes.length & 0xFF);
            out.write(modeBytes);
            out.writeInt(durationSeconds);
            out.flush();
            cam.sendPluginMessage(this, SESSION_CHANNEL, baos.toByteArray());
        } catch (IOException e) {
            getLogger().warning("Failed to build session_start payload: " + e.getMessage());
        }
    }

    // ----------------------------------------------------------- follow task

    private void startFollowingTask() {
        stopFollowingTask();
        followTask = Bukkit.getScheduler().runTaskTimer(this, () -> {
            Player c = controller;
            Player k = camera;
            if (c == null || k == null || !c.isOnline() || !k.isOnline()) return;

            Location cLoc = c.getLocation();
            Location camLoc = k.getLocation();

            boolean worldChanged = camLoc.getWorld() != cLoc.getWorld();
            boolean moved = worldChanged || camLoc.distanceSquared(cLoc) > 0.0025;
            boolean rotated = Math.abs(camLoc.getYaw() - cLoc.getYaw()) > 0.5f
                    || Math.abs(camLoc.getPitch() - cLoc.getPitch()) > 0.5f;

            if (moved || rotated) {
                // teleportAsync keeps yaw+pitch -> the camera's look mirrors the controller.
                k.teleportAsync(cLoc.clone());
            }
        }, 0L, 1L);
    }

    private void stopFollowingTask() {
        if (followTask != null) {
            followTask.cancel();
            followTask = null;
        }
    }

    // --------------------------------------------------------- camera physics

    private void prepareCamera(Player cam) {
        cam.setGameMode(GameMode.SURVIVAL);   // a real survival body, but frozen/observer-like
        cam.setGravity(false);
        cam.setVelocity(new Vector(0, 0, 0));
        cam.setAllowFlight(true);
        cam.setFlying(true);
        cam.setInvulnerable(true);
        cam.setCollidable(false);
    }

    private void restoreCamera(Player cam) {
        if (cam == null || !cam.isOnline()) return;
        cam.setGravity(true);
        cam.setInvulnerable(false);
        cam.setCollidable(true);
        if (cam.getGameMode() != GameMode.CREATIVE && cam.getGameMode() != GameMode.SPECTATOR) {
            cam.setAllowFlight(false);
            if (cam.isFlying()) cam.setFlying(false);
        }
        if (controller != null && controller.isOnline()) {
            controller.showPlayer(this, cam);
            cam.showPlayer(this, controller);
        }
    }
}
