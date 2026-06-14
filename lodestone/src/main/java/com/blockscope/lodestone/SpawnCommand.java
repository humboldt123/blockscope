package com.blockscope.lodestone;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Op-only in-game tool for curating spawn points in the data-collection world library.
 *
 *   /spawn list            — count + current index + ~10 entries
 *   /spawn go <n>          — teleport to index n
 *   /spawn next | prev     — cycle the per-player current index
 *   /spawn random          — teleport to a random spawn (also: /spoofspawn)
 *   /spawn add             — add the player's exact position as a "manual" SpawnPoint
 *   /spawn del [<n>]       — delete index n, or the nearest spawn if no index
 *   /spawn save            — force-write the source file
 *   /spawn reload          — reload the list from disk
 *
 * All edits operate on the SOURCE map (map_library/&lt;source&gt;/spawn_candidates.json)
 * that the player's current copy world was made from, and update the shared in-memory
 * list so the live session reflects the change immediately.
 */
public class SpawnCommand implements CommandExecutor, TabCompleter {

    private static final List<String> SUBCOMMANDS = List.of(
        "list", "go", "next", "prev", "random", "add", "del", "save", "reload");

    private final SessionManager sessions;
    private final org.bukkit.plugin.Plugin plugin;
    // Per-player current spawn index for go/next/prev.
    private final Map<UUID, Integer> currentIndex = new ConcurrentHashMap<>();

    public SpawnCommand(org.bukkit.plugin.Plugin plugin, SessionManager sessions) {
        this.plugin = plugin;
        this.sessions = sessions;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage("This command is player-only.");
            return true;
        }
        if (!sender.isOp()) {
            player.sendMessage("§cYou must be an operator to use this command.");
            return true;
        }

        // /spoofspawn → /spawn random
        if (command.getName().equalsIgnoreCase("spoofspawn")) {
            return doRandom(player);
        }

        String sub = (args.length == 0) ? "list" : args[0].toLowerCase();
        switch (sub) {
            case "list":   return doList(player);
            case "go":     return doGo(player, args);
            case "next":   return doCycle(player, +1);
            case "prev":   return doCycle(player, -1);
            case "random": return doRandom(player);
            case "add":    return doAdd(player);
            case "del":
            case "delete": return doDel(player, args);
            case "save":   return doSave(player);
            case "reload": return doReload(player);
            default:
                player.sendMessage("§eUsage: /spawn <list|go <n>|next|prev|random|add|del [n]|save|reload>");
                return true;
        }
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        // Only /spawn has subcommands; /spoofspawn takes no args.
        if (command.getName().equalsIgnoreCase("spawn") && args.length == 1) {
            String prefix = args[0].toLowerCase();
            List<String> out = new ArrayList<>();
            for (String s : SUBCOMMANDS) {
                if (s.startsWith(prefix)) out.add(s);
            }
            return out;
        }
        return Collections.emptyList();
    }

    // ── Context resolution ──────────────────────────────────────────────────────

    private String sourceFor(Player player) {
        return sessions.getLibrarySource(player.getWorld().getName());
    }

    /** Returns the live spawn list for the player's world, or null with an error message. */
    private List<SpawnPoint> requireList(Player player) {
        String source = sourceFor(player);
        if (source == null) {
            player.sendMessage("§cYou are not in a library world — spawn editing is unavailable here.");
            return null;
        }
        List<SpawnPoint> list = sessions.getSpawnsForWorld(player.getWorld().getName());
        if (list == null) {
            player.sendMessage("§cNo spawn list is loaded for this world (source: " + source + ").");
            return null;
        }
        return list;
    }

    // ── Subcommands ─────────────────────────────────────────────────────────────

    private boolean doList(Player player) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        String source = sourceFor(player);
        int cur = currentIndex.getOrDefault(player.getUniqueId(), 0);
        synchronized (list) {
            player.sendMessage("§e[Blockscope] " + source + ": " + list.size()
                + " spawns, current index = " + cur);
            int start = Math.max(0, Math.min(cur - 2, list.size() - 10));
            if (start < 0) start = 0;
            int end = Math.min(list.size(), start + 10);
            for (int i = start; i < end; i++) {
                SpawnPoint p = list.get(i);
                String marker = (i == cur) ? "§a> " : "§7  ";
                player.sendMessage(marker + "[" + i + "] " + p.label + " @ " + p.x + "," + p.y + "," + p.z);
            }
        }
        return true;
    }

    private boolean doGo(Player player, String[] args) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        if (args.length < 2) { player.sendMessage("§eUsage: /spawn go <n>"); return true; }
        int n;
        try { n = Integer.parseInt(args[1]); }
        catch (NumberFormatException e) { player.sendMessage("§cNot a number: " + args[1]); return true; }
        SpawnPoint p;
        synchronized (list) {
            if (n < 0 || n >= list.size()) {
                player.sendMessage("§cIndex out of range (0.." + (list.size() - 1) + ").");
                return true;
            }
            p = list.get(n);
        }
        currentIndex.put(player.getUniqueId(), n);
        teleport(player, p, n);
        return true;
    }

    private boolean doCycle(Player player, int delta) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        SpawnPoint p;
        int idx;
        synchronized (list) {
            if (list.isEmpty()) { player.sendMessage("§cNo spawns to cycle."); return true; }
            int cur = currentIndex.getOrDefault(player.getUniqueId(), 0);
            int size = list.size();
            idx = ((cur + delta) % size + size) % size;
            p = list.get(idx);
        }
        currentIndex.put(player.getUniqueId(), idx);
        teleport(player, p, idx);
        return true;
    }

    private boolean doRandom(Player player) {
        if (!player.isOp()) { player.sendMessage("§cYou must be an operator to use this command."); return true; }
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        SpawnPoint p;
        int idx;
        synchronized (list) {
            if (list.isEmpty()) { player.sendMessage("§cNo spawns available."); return true; }
            idx = (int) (Math.random() * list.size());
            p = list.get(idx);
        }
        currentIndex.put(player.getUniqueId(), idx);
        teleport(player, p, idx);
        return true;
    }

    private boolean doAdd(Player player) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        Location loc = player.getLocation();
        // Exact current position — the user chose where to stand. Do NOT ground-snap.
        SpawnPoint p = new SpawnPoint(
            (int) Math.floor(loc.getX()),
            (int) Math.floor(loc.getY()),
            (int) Math.floor(loc.getZ()),
            "manual");
        int idx;
        synchronized (list) {
            list.add(p);
            idx = list.size() - 1;
        }
        currentIndex.put(player.getUniqueId(), idx);
        if (persist(player, list)) {
            player.sendMessage("§aAdded [" + idx + "] " + p.label + " @ " + p.x + "," + p.y + "," + p.z
                + " (saved).");
        }
        return true;
    }

    private boolean doDel(Player player, String[] args) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        SpawnPoint removed;
        int idx;
        synchronized (list) {
            if (list.isEmpty()) { player.sendMessage("§cNo spawns to delete."); return true; }
            if (args.length >= 2) {
                try { idx = Integer.parseInt(args[1]); }
                catch (NumberFormatException e) { player.sendMessage("§cNot a number: " + args[1]); return true; }
                if (idx < 0 || idx >= list.size()) {
                    player.sendMessage("§cIndex out of range (0.." + (list.size() - 1) + ").");
                    return true;
                }
            } else {
                idx = nearestIndex(list, player.getLocation());
            }
            removed = list.remove(idx);
        }
        if (persist(player, list)) {
            player.sendMessage("§aDeleted [" + idx + "] " + removed.label
                + " @ " + removed.x + "," + removed.y + "," + removed.z + " (saved).");
        }
        return true;
    }

    private boolean doSave(Player player) {
        List<SpawnPoint> list = requireList(player);
        if (list == null) return true;
        if (persist(player, list)) {
            int size;
            synchronized (list) { size = list.size(); }
            player.sendMessage("§aSaved " + size + " spawns to " + sourceFor(player) + ".");
        }
        return true;
    }

    private boolean doReload(Player player) {
        String source = sourceFor(player);
        if (source == null) {
            player.sendMessage("§cYou are not in a library world — spawn editing is unavailable here.");
            return true;
        }
        int size = sessions.reloadSpawnsForSource(source);
        currentIndex.remove(player.getUniqueId());
        player.sendMessage("§aReloaded " + size + " spawns from " + source + ".");
        return true;
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────

    private int nearestIndex(List<SpawnPoint> list, Location loc) {
        int best = 0;
        double bestD = Double.MAX_VALUE;
        for (int i = 0; i < list.size(); i++) {
            SpawnPoint p = list.get(i);
            double dx = (p.x + 0.5) - loc.getX();
            double dy = p.y - loc.getY();
            double dz = (p.z + 0.5) - loc.getZ();
            double d = dx * dx + dy * dy + dz * dz;
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    private void teleport(Player player, SpawnPoint p, int idx) {
        World w = player.getWorld();
        player.teleport(SessionManager.toLocation(w, p));
        player.sendMessage("§e[" + idx + "] " + p.label + " @ " + p.x + "," + p.y + "," + p.z);
    }

    /** Persist the source file; returns true on success, messages the player on failure. */
    private boolean persist(Player player, List<SpawnPoint> list) {
        String source = sourceFor(player);
        if (source == null) {
            player.sendMessage("§cCannot save: not in a library world.");
            return false;
        }
        try {
            sessions.saveSpawnCandidates(source, list);
            return true;
        } catch (IOException e) {
            player.sendMessage("§cSave failed: " + e.getMessage());
            plugin.getLogger().warning("Failed to save spawn_candidates.json for " + source + ": " + e.getMessage());
            return false;
        }
    }
}
