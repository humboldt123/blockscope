package com.blockscope.lodestone;

import org.bukkit.Bukkit;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitTask;
import org.bukkit.util.Vector;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Paired (controller, camera) session support — purely additive on top of the existing
 * single-player {@link SessionManager} flow. Nothing here touches the world pool,
 * copy-on-join, spawn candidates, or the existing session machinery.
 *
 * <p>Pairing is by username convention: a Mineflayer controller bot connects as
 * {@code ctrl_<id>} and its record-only Fabric camera connects as {@code cam_<id>}.
 * Lodestone detects the prefix on join:
 * <ul>
 *   <li>{@code ctrl_<id>} → goes through the normal {@link SessionManager} join flow
 *       (gets a pooled world, gamemode, tools, etc.). We additionally remember the
 *       world assignment under {@code <id>} so the camera can find it. The Mineflayer
 *       bot ignores plugin messages and drives itself.</li>
 *   <li>{@code cam_<id>} → does NOT go through the normal flow. It is teleported into the
 *       SAME world already assigned to its controller, made an ADVENTURE free-floating
 *       observer (no gravity, flying, non-colliding), hidden from / blind to the
 *       controller, and starts the per-tick teleport-mirror task. It is sent a
 *       {@code blockscope:session_start} (mode "adventure") so the Fabric mod records.</li>
 * </ul>
 *
 * <p>Session end: when EITHER member of a pair disconnects, the mirror task is cancelled
 * and {@link SessionManager#endSessionForPair} tears down BOTH players' worlds via the
 * existing endSession path (so {@code playerWorlds} is cleaned up and the world is
 * unloaded/deleted exactly as for a single-player session).
 *
 * <p>Any non-prefixed connection falls through untouched to the existing assignMode flow.
 */
public class MirrorPairManager {

    static final String CTRL_PREFIX = "ctrl_";
    static final String CAM_PREFIX  = "cam_";
    private static final String SESSION_CHANNEL = "blockscope:session_start";

    private final LodestonePlugin plugin;
    private final SessionManager sessions;

    /** Pair id → controller world (populated when ctrl_ joins, read when cam_ joins). */
    private final Map<String, World> pairWorlds = new ConcurrentHashMap<>();
    /** Pair id → controller UUID. */
    private final Map<String, UUID> controllerOf = new ConcurrentHashMap<>();
    /** Pair id → camera UUID. */
    private final Map<String, UUID> cameraOf = new ConcurrentHashMap<>();
    /** Pair id → running mirror follow task. */
    private final Map<String, BukkitTask> followTasks = new ConcurrentHashMap<>();
    /** Manual /mirrorpair overrides: cameraUUID → controllerUUID (for the debug command). */
    private final Map<UUID, UUID> manualHidden = new ConcurrentHashMap<>();

    public MirrorPairManager(LodestonePlugin plugin, SessionManager sessions) {
        this.plugin = plugin;
        this.sessions = sessions;
    }

    // ── Naming helpers ──────────────────────────────────────────────────────────

    static boolean isController(String name) { return name != null && name.startsWith(CTRL_PREFIX); }
    static boolean isCamera(String name)     { return name != null && name.startsWith(CAM_PREFIX); }
    static boolean isPaired(String name)     { return isController(name) || isCamera(name); }

    private static String pairIdOf(String name) {
        if (isController(name)) return name.substring(CTRL_PREFIX.length());
        if (isCamera(name))     return name.substring(CAM_PREFIX.length());
        return null;
    }

    // ── Join hooks (called from SessionManager.onPlayerJoin) ─────────────────────

    /**
     * Called after the controller has been assigned a pooled world by the normal flow.
     * Records the world so the camera (joining shortly after) lands in the SAME world.
     */
    void onControllerAssigned(Player controller, World world) {
        String id = pairIdOf(controller.getName());
        if (id == null) return;
        pairWorlds.put(id, world);
        controllerOf.put(id, controller.getUniqueId());
        plugin.getLogger().info("[pair] controller " + controller.getName()
            + " assigned world " + world.getName() + " (pair id=" + id + ")");
    }

    /**
     * Called from onPlayerJoin for a {@code cam_<id>} player INSTEAD of the normal flow.
     * Teleports the camera into the controller's world and starts the mirror. If the
     * controller's world isn't known yet (camera joined first / controller already gone),
     * retries for a few seconds, then kicks the camera so it reconnects cleanly.
     */
    void onCameraJoin(Player camera) {
        String id = pairIdOf(camera.getName());
        if (id == null) return;
        cameraOf.put(id, camera.getUniqueId());
        plugin.getLogger().info("[pair] camera " + camera.getName() + " joined (pair id=" + id + "); awaiting controller world ...");
        tryStartMirror(camera, id, 0);
    }

    /** Poll for the controller's world (it may join slightly after the camera) up to ~15s. */
    private void tryStartMirror(Player camera, String id, int attempt) {
        if (!camera.isOnline()) return;
        World world = pairWorlds.get(id);
        UUID ctrlUuid = controllerOf.get(id);
        Player controller = (ctrlUuid != null) ? Bukkit.getPlayer(ctrlUuid) : null;

        if (world != null && controller != null && controller.isOnline()) {
            startPairedSession(controller, camera, id, world);
            return;
        }
        if (attempt >= 60) { // 60 * 250ms = 15s
            plugin.getLogger().warning("[pair] camera " + camera.getName()
                + " never found its controller — kicking to reconnect.");
            camera.kick(net.kyori.adventure.text.Component.text("§eWaiting for controller — reconnecting…"));
            return;
        }
        Bukkit.getScheduler().runTaskLater(plugin, () -> tryStartMirror(camera, id, attempt + 1), 5L);
    }

    /**
     * Move the camera into the controller's world, make it an observer, hide the pair from
     * each other, start the follow task, and send session_start so the Fabric mod records.
     */
    private void startPairedSession(Player controller, Player camera, String id, World world) {
        // Register the camera's world with the SessionManager so its damage/respawn guards
        // and endSession teardown treat it exactly like any other session player.
        sessions.registerPairCamera(camera, world);

        camera.getInventory().clear();
        camera.setGameMode(GameMode.ADVENTURE);
        camera.setGravity(false);
        camera.setVelocity(new Vector(0, 0, 0));
        camera.setAllowFlight(true);
        camera.setFlying(true);
        camera.setInvulnerable(true);
        camera.setCollidable(false);

        // Snap onto the controller immediately (this is what moves the camera into the
        // controller's world), then start the per-tick follow.
        camera.teleport(controller.getLocation().clone());

        // Mutual invisibility so neither appears in the other's view / recording.
        controller.hidePlayer(plugin, camera);
        camera.hidePlayer(plugin, controller);

        startFollowTask(id, controller, camera);

        plugin.getLogger().info("[pair] mirror started: controller=" + controller.getName()
            + " → camera=" + camera.getName() + " in " + world.getName());

        // Send session_start (mode "adventure") so the record-only Fabric mod begins recording.
        Bukkit.getScheduler().runTaskLater(plugin, () -> {
            if (camera.isOnline()) {
                sendSessionStart(camera);
                plugin.getLogger().info("[pair] sent session_start → " + camera.getName() + " [adventure]");
            }
        }, 20L);
    }

    private void startFollowTask(String id, Player controller, Player camera) {
        BukkitTask old = followTasks.remove(id);
        if (old != null) old.cancel();

        final UUID cUuid = controller.getUniqueId();
        final UUID kUuid = camera.getUniqueId();
        BukkitTask task = Bukkit.getScheduler().runTaskTimer(plugin, () -> {
            Player c = Bukkit.getPlayer(cUuid);
            Player k = Bukkit.getPlayer(kUuid);
            if (c == null || k == null || !c.isOnline() || !k.isOnline()) return;

            Location cLoc = c.getLocation();
            Location camLoc = k.getLocation();
            boolean worldChanged = camLoc.getWorld() != cLoc.getWorld();
            boolean moved = worldChanged || camLoc.distanceSquared(cLoc) > 0.0025;
            boolean rotated = Math.abs(camLoc.getYaw() - cLoc.getYaw()) > 0.5f
                || Math.abs(camLoc.getPitch() - cLoc.getPitch()) > 0.5f;
            if (moved || rotated) {
                k.teleportAsync(cLoc.clone());
            }
        }, 0L, 1L);
        followTasks.put(id, task);
    }

    // ── Quit hook (called from SessionManager.onPlayerQuit) ──────────────────────

    /**
     * Called when a {@code ctrl_*}/{@code cam_*} player quits. Cancels the mirror task and
     * returns the partner Player (if still online) so the SessionManager can end its session
     * too. Returns null if there's no online partner to tear down. Also cleans up pair state.
     */
    Player onPairedQuit(Player player) {
        String id = pairIdOf(player.getName());
        if (id == null) return null;

        BukkitTask task = followTasks.remove(id);
        if (task != null) task.cancel();

        pairWorlds.remove(id);
        UUID ctrlUuid = controllerOf.remove(id);
        UUID camUuid = cameraOf.remove(id);

        // Find the partner (the other member of the pair) that is still online.
        UUID partnerUuid = player.getUniqueId().equals(ctrlUuid) ? camUuid : ctrlUuid;
        Player partner = (partnerUuid != null) ? Bukkit.getPlayer(partnerUuid) : null;

        plugin.getLogger().info("[pair] " + player.getName() + " quit; tearing down pair id=" + id
            + (partner != null ? " (partner " + partner.getName() + " online)" : ""));
        return (partner != null && partner.isOnline()) ? partner : null;
    }

    // ── /mirrorpair debug command ────────────────────────────────────────────────

    /** Manually mirror any two online players regardless of naming (op debug override). */
    boolean handleMirrorPair(org.bukkit.command.CommandSender sender, String[] args) {
        if (args.length == 1 && args[0].equalsIgnoreCase("stop")) {
            for (BukkitTask t : followTasks.values()) t.cancel();
            followTasks.clear();
            sender.sendMessage("§e[mirrorpair] all manual mirror tasks stopped.");
            return true;
        }
        if (args.length != 2) {
            sender.sendMessage("§e[mirrorpair] usage: /mirrorpair <controller> <camera> | /mirrorpair stop");
            return true;
        }
        Player c = Bukkit.getPlayerExact(args[0]);
        Player k = Bukkit.getPlayerExact(args[1]);
        if (c == null) { sender.sendMessage("§c[mirrorpair] controller '" + args[0] + "' not online."); return true; }
        if (k == null) { sender.sendMessage("§c[mirrorpair] camera '" + args[1] + "' not online."); return true; }
        if (c.equals(k)) { sender.sendMessage("§c[mirrorpair] controller and camera must differ."); return true; }

        String id = "manual_" + c.getName() + "_" + k.getName();
        k.setGameMode(GameMode.ADVENTURE);
        k.setGravity(false);
        k.setVelocity(new Vector(0, 0, 0));
        k.setAllowFlight(true);
        k.setFlying(true);
        k.setInvulnerable(true);
        k.setCollidable(false);
        k.teleport(c.getLocation().clone());
        c.hidePlayer(plugin, k);
        k.hidePlayer(plugin, c);
        startFollowTask(id, c, k);
        sendSessionStart(k);
        sender.sendMessage("§a[mirrorpair] " + c.getName() + " → camera " + k.getName()
            + " (following + recording).");
        plugin.getLogger().info("[pair] manual mirrorpair controller=" + c.getName() + " camera=" + k.getName());
        return true;
    }

    // ── session_start packet (same wire format SessionManager uses) ──────────────

    private void sendSessionStart(Player camera) {
        String mode = "adventure";
        byte[] modeBytes = mode.getBytes(StandardCharsets.UTF_8);
        ByteBuffer buf = ByteBuffer.allocate(1 + modeBytes.length + 4);
        buf.put((byte) modeBytes.length);
        buf.put(modeBytes);
        buf.putInt(sessions.getSessionDurationSeconds());
        camera.sendPluginMessage(plugin, SESSION_CHANNEL, buf.array());
    }
}
