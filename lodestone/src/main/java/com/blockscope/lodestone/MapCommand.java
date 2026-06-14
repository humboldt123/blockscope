package com.blockscope.lodestone;

import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Op-only inspection tools for the data-collection world library.
 *
 *   /maps             — list every discovered library source + spawn-candidate count.
 *   /visit <source>   — serve a fresh copy of that exact source map and teleport in
 *                       (CREATIVE/fly) so the op can curate spawns with /spawn add etc.
 *
 * A /visit world persists until the op disconnects or runs /visit again; the previous
 * visit/session world is torn down first (no leaked worlds). No bot session is started.
 */
public class MapCommand implements CommandExecutor, TabCompleter {

    private final SessionManager sessions;

    public MapCommand(SessionManager sessions) {
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

        if (command.getName().equalsIgnoreCase("maps")) {
            return doMaps(player);
        }
        return doVisit(player, args);
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        // /maps takes no args; /visit <source> completes from discovered source names.
        if (command.getName().equalsIgnoreCase("visit") && args.length == 1) {
            String prefix = args[0].toLowerCase();
            List<String> out = new ArrayList<>();
            for (String name : sessions.librarySourceNames()) {
                if (name.toLowerCase().startsWith(prefix)) out.add(name);
            }
            return out;
        }
        return Collections.emptyList();
    }

    private boolean doMaps(Player player) {
        List<Map.Entry<String, Integer>> sources = sessions.listLibrarySources();
        if (sources.isEmpty()) {
            player.sendMessage("§cNo library sources discovered.");
            return true;
        }
        player.sendMessage("§e[Blockscope] " + sources.size() + " library sources:");
        for (Map.Entry<String, Integer> e : sources) {
            player.sendMessage("§7  " + e.getKey() + " §8(" + e.getValue() + ")");
        }
        return true;
    }

    private boolean doVisit(Player player, String[] args) {
        if (args.length < 1) {
            player.sendMessage("§eUsage: /visit <source>  (see /maps for valid names)");
            return true;
        }
        // Allow source names with spaces, just in case.
        String query = String.join(" ", args).trim();
        File source = sessions.resolveSource(query);
        if (source == null) {
            player.sendMessage("§cUnknown or ambiguous source: '" + query + "'. Run /maps for valid names.");
            return true;
        }
        player.sendMessage("§eCopying '" + source.getName()
            + "' — instant for small maps, up to ~30s for large ones (e.g. hermitcraft). "
            + "You'll be teleported in when it's ready…");
        sessions.visitSource(player, source);
        return true;
    }
}
