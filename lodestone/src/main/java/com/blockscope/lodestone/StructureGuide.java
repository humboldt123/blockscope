package com.blockscope.lodestone;

import org.bukkit.Location;
import org.bukkit.StructureType;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.nio.ByteBuffer;
import java.util.*;

/**
 * Periodically locates nearby Minecraft structures and broadcasts their block
 * positions to the client mod via the "blockscope:structures" plugin channel.
 *
 * The client mod (BotModule) queues received positions as priority exploration
 * goals.  The channel is optional — the client works without it, but loses
 * server-assisted structure targeting.
 *
 * Scans run synchronously on the main thread (world access is not thread-safe)
 * but only once every SCAN_INTERVAL_TICKS per player and rotate through one
 * structure type at a time to spread the cost.
 */
public class StructureGuide implements Listener {

    static final String CHANNEL = "blockscope:structures";

    private static final int SCAN_INTERVAL_TICKS = 400; // ~20 s between scans per player
    private static final int LOCATE_RADIUS = 128;       // chunks; ~2048 blocks

    // Overworld structure types most likely to contain interior spaces worth recording.
    private static final StructureType[] STRUCTURE_TYPES = {
        StructureType.VILLAGE,
        StructureType.PILLAGER_OUTPOST,
        StructureType.STRONGHOLD,
        StructureType.MINESHAFT,
        StructureType.JUNGLE_PYRAMID,
        StructureType.DESERT_PYRAMID,
        StructureType.WOODLAND_MANSION,
        StructureType.SWAMP_HUT,
        StructureType.OCEAN_RUIN,
        StructureType.SHIPWRECK,
    };

    private final JavaPlugin plugin;
    // Per-player: last scan tick and which structure type to scan next
    private final Map<UUID, Long> lastScanTick = new HashMap<>();
    private final Map<UUID, Integer> nextTypeIdx = new HashMap<>();

    public StructureGuide(JavaPlugin plugin) {
        this.plugin = plugin;
        plugin.getServer().getMessenger().registerOutgoingPluginChannel(plugin, CHANNEL);
    }

    @EventHandler
    public void onPlayerMove(PlayerMoveEvent event) {
        // Only act on block-boundary crossings to avoid firing every sub-block movement
        if (event.getFrom().getBlockX() == event.getTo().getBlockX()
                && event.getFrom().getBlockZ() == event.getTo().getBlockZ()) return;

        Player player = event.getPlayer();
        UUID uid = player.getUniqueId();
        long tick = plugin.getServer().getCurrentTick();

        if (tick - lastScanTick.getOrDefault(uid, 0L) < SCAN_INTERVAL_TICKS) return;
        lastScanTick.put(uid, tick);

        // Rotate through structure types one per scan to spread main-thread cost
        int idx = nextTypeIdx.getOrDefault(uid, 0);
        nextTypeIdx.put(uid, (idx + 1) % STRUCTURE_TYPES.length);
        StructureType type = STRUCTURE_TYPES[idx];

        Location loc;
        try {
            loc = player.getWorld().locateNearestStructure(
                player.getLocation(), type, LOCATE_RADIUS, false);
        } catch (Exception e) {
            // Some structure types aren't valid in certain dimensions; skip silently
            return;
        }
        if (loc == null) return;

        byte[] payload = ByteBuffer.allocate(12)
            .putInt(1)
            .putInt(loc.getBlockX())
            .putInt(loc.getBlockZ())
            .array();
        player.sendPluginMessage(plugin, CHANNEL, payload);
    }
}
