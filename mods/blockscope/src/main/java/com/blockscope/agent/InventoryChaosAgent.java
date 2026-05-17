package com.blockscope.agent;

import net.minecraft.block.Blocks;
import net.minecraft.block.entity.ChestBlockEntity;
import net.minecraft.block.entity.ShulkerBoxBlockEntity;
import net.minecraft.client.MinecraftClient;
import net.minecraft.entity.ItemEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.screen.slot.SlotActionType;
import net.minecraft.text.Text;
import net.minecraft.util.Hand;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.util.math.Direction;
import net.minecraft.util.math.Vec3d;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

public class InventoryChaosAgent {
    private static InventoryChaosAgent instance;
    private boolean running = false;
    private Thread agentThread;
    private final Set<BlockPos> visitedChests = new HashSet<>();
    private final Random random = new Random();

    private enum Action {
        CHEST_LOOT_SWAP,
        CHEST_BREAK_COLLECT,
        ITEM_THROW,
        PICKUP_WALK,
        HOTBAR_SHUFFLE,
        SHULKER_RUMMAGE,
        BARITONE_DROP_ITEMS
    }

    private InventoryChaosAgent() {}

    public static InventoryChaosAgent getInstance() {
        if (instance == null) {
            instance = new InventoryChaosAgent();
        }
        return instance;
    }

    public void toggle() {
        if (running) {
            stop();
        } else {
            start();
        }
    }

    public void start() {
        if (running) return;

        running = true;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player != null) {
            client.player.sendMessage(Text.literal("§a[Chaos Agent] Starting..."), false);
        }

        agentThread = new Thread(this::agentLoop, "InventoryChaosAgent");
        agentThread.setDaemon(true);
        agentThread.start();
    }

    public void stop() {
        if (!running) return;

        running = false;
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player != null) {
            client.player.sendMessage(Text.literal("§c[Chaos Agent] Stopped."), false);
        }

        if (agentThread != null) {
            agentThread.interrupt();
            agentThread = null;
        }
    }

    public boolean isRunning() {
        return running;
    }

    private void agentLoop() {
        MinecraftClient client = MinecraftClient.getInstance();

        while (running && !Thread.currentThread().isInterrupted()) {
            try {
                if (client.player == null || client.world == null) {
                    Thread.sleep(1000);
                    continue;
                }

                // Pick random action
                Action action = Action.values()[random.nextInt(Action.values().length)];

                // Execute action
                executeAction(action, client);

                // Wait 1-3 seconds between actions
                Thread.sleep(1000 + random.nextInt(2000));

            } catch (InterruptedException e) {
                break;
            } catch (Exception e) {
                System.err.println("[Chaos Agent] Error: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private void executeAction(Action action, MinecraftClient client) throws InterruptedException {
        switch (action) {
            case CHEST_LOOT_SWAP:
                chestLootSwap(client);
                break;
            case CHEST_BREAK_COLLECT:
                chestBreakCollect(client);
                break;
            case ITEM_THROW:
                itemThrow(client);
                break;
            case PICKUP_WALK:
                pickupWalk(client);
                break;
            case HOTBAR_SHUFFLE:
                hotbarShuffle(client);
                break;
            case SHULKER_RUMMAGE:
                shulkerRummage(client);
                break;
            case BARITONE_DROP_ITEMS:
                baritoneDropItems(client);
                break;
        }
    }

    private void chestLootSwap(MinecraftClient client) throws InterruptedException {
        BlockPos chestPos = findNearestChest(client, false);
        if (chestPos == null) return;

        log(client, "§eNavigating to chest at " + chestPos.toShortString());
        if (!navigateToBlock(client, chestPos)) return;

        log(client, "§eOpening chest and swapping items...");
        client.execute(() -> {
            client.interactionManager.interactBlock(
                client.player,
                Hand.MAIN_HAND,
                new BlockHitResult(Vec3d.ofCenter(chestPos), Direction.UP, chestPos, false)
            );
        });

        Thread.sleep(500);

        // Swap items
        if (client.player.currentScreenHandler != null) {
            for (int i = 0; i < 3 + random.nextInt(5); i++) {
                int slot = random.nextInt(client.player.currentScreenHandler.slots.size());
                client.execute(() -> {
                    client.interactionManager.clickSlot(
                        client.player.currentScreenHandler.syncId,
                        slot,
                        0,
                        SlotActionType.QUICK_MOVE,
                        client.player
                    );
                });
                Thread.sleep(100);
            }

            client.execute(() -> client.player.closeHandledScreen());
        }

        visitedChests.add(chestPos);
    }

    private void chestBreakCollect(MinecraftClient client) throws InterruptedException {
        BlockPos chestPos = findNearestChest(client, false);
        if (chestPos == null) return;

        log(client, "§eNavigating to chest to break at " + chestPos.toShortString());
        if (!navigateToBlock(client, chestPos)) return;

        log(client, "§eBreaking chest...");
        client.execute(() -> {
            client.interactionManager.attackBlock(chestPos, Direction.UP);
            client.interactionManager.breakBlock(chestPos);
        });

        Thread.sleep(1000);
        log(client, "§eCollecting dropped items...");
    }

    private void itemThrow(MinecraftClient client) throws InterruptedException {
        log(client, "§eThrowing items from inventory...");

        int itemsToThrow = 1 + random.nextInt(3);
        for (int i = 0; i < itemsToThrow; i++) {
            int slot = random.nextInt(36);
            ItemStack stack = client.player.getInventory().getStack(slot);

            if (!stack.isEmpty()) {
                client.execute(() -> client.player.dropItem(stack.copy(), false));
                Thread.sleep(200);
            }
        }
    }

    private void pickupWalk(MinecraftClient client) throws InterruptedException {
        BlockPos playerPos = client.player.getBlockPos();
        List<ItemEntity> nearbyItems = client.world.getEntitiesByClass(
            ItemEntity.class,
            new Box(playerPos).expand(16.0),
            item -> true
        );

        if (nearbyItems.isEmpty()) {
            log(client, "§eNo items nearby to pickup");
            return;
        }

        ItemEntity target = nearbyItems.get(random.nextInt(nearbyItems.size()));
        BlockPos itemPos = target.getBlockPos();

        log(client, "§eWalking to pickup items at " + itemPos.toShortString());
        navigateToBlock(client, itemPos);
    }

    private void hotbarShuffle(MinecraftClient client) throws InterruptedException {
        log(client, "§eShuffling hotbar...");

        for (int i = 0; i < 5 + random.nextInt(10); i++) {
            int slot1 = random.nextInt(9);
            int slot2 = 9 + random.nextInt(27);

            client.execute(() -> {
                client.interactionManager.clickSlot(
                    client.player.currentScreenHandler.syncId,
                    slot1,
                    slot2,
                    SlotActionType.SWAP,
                    client.player
                );
            });

            Thread.sleep(100);
        }
    }

    private void shulkerRummage(MinecraftClient client) throws InterruptedException {
        BlockPos shulkerPos = findNearestShulkerBox(client);
        if (shulkerPos == null) {
            log(client, "§eNo shulker boxes nearby");
            return;
        }

        log(client, "§eNavigating to shulker box at " + shulkerPos.toShortString());
        if (!navigateToBlock(client, shulkerPos)) return;

        log(client, "§eRummaging through shulker box...");
        client.execute(() -> {
            client.interactionManager.interactBlock(
                client.player,
                Hand.MAIN_HAND,
                new BlockHitResult(Vec3d.ofCenter(shulkerPos), Direction.UP, shulkerPos, false)
            );
        });

        Thread.sleep(500);

        if (client.player.currentScreenHandler != null) {
            for (int i = 0; i < 3 + random.nextInt(5); i++) {
                int slot = random.nextInt(client.player.currentScreenHandler.slots.size());
                client.execute(() -> {
                    client.interactionManager.clickSlot(
                        client.player.currentScreenHandler.syncId,
                        slot,
                        0,
                        SlotActionType.QUICK_MOVE,
                        client.player
                    );
                });
                Thread.sleep(100);
            }

            client.execute(() -> client.player.closeHandledScreen());
        }
    }

    private void baritoneDropItems(MinecraftClient client) throws InterruptedException {
        log(client, "§eBaritone: Dropping and swapping items...");

        // Drop items
        for (int i = 0; i < 2 + random.nextInt(3); i++) {
            int slot = random.nextInt(36);
            ItemStack stack = client.player.getInventory().getStack(slot);

            if (!stack.isEmpty()) {
                client.execute(() -> {
                    client.player.dropItem(stack.copy(), true);
                });
                Thread.sleep(300);
            }
        }

        // Swap hotbar slots
        Thread.sleep(500);
        for (int i = 0; i < 3; i++) {
            int newSlot = random.nextInt(9);
            client.execute(() -> client.player.getInventory().selectedSlot = newSlot);
            Thread.sleep(200);
        }
    }

    private boolean navigateToBlock(MinecraftClient client, BlockPos pos) throws InterruptedException {
        try {
            // Use Baritone if available (reflection-based for optional dependency)
            Class<?> baritoneAPI = Class.forName("baritone.api.BaritoneAPI");
            Class<?> goalNearClass = Class.forName("baritone.api.pathing.goals.GoalNear");

            Object provider = baritoneAPI.getMethod("getProvider").invoke(null);
            Object baritone = provider.getClass().getMethod("getPrimaryBaritone").invoke(provider);
            Object goalProcess = baritone.getClass().getMethod("getCustomGoalProcess").invoke(baritone);

            Object goalNear = goalNearClass.getConstructor(BlockPos.class, int.class).newInstance(pos, 2);
            goalProcess.getClass().getMethod("setGoalAndPath", Class.forName("baritone.api.pathing.goals.Goal")).invoke(goalProcess, goalNear);

            // Wait up to 15 seconds for Baritone to reach goal
            int timeout = 150;
            while (timeout > 0 && running) {
                boolean isActive = (boolean) goalProcess.getClass().getMethod("isActive").invoke(goalProcess);
                if (!isActive || client.player.getBlockPos().isWithinDistance(pos, 3)) {
                    return true;
                }
                Thread.sleep(100);
                timeout--;
            }

            return client.player.getBlockPos().isWithinDistance(pos, 3);

        } catch (Exception e) {
            // Baritone not available, log and return false
            log(client, "§cBaritone not found! Please install Baritone for navigation.");
            return false;
        }
    }

    private BlockPos findNearestChest(MinecraftClient client, boolean excludeVisited) {
        BlockPos playerPos = client.player.getBlockPos();
        BlockPos nearestChest = null;
        double nearestDistance = Double.MAX_VALUE;

        int range = 32;
        for (int x = -range; x <= range; x++) {
            for (int y = -range; y <= range; y++) {
                for (int z = -range; z <= range; z++) {
                    BlockPos pos = playerPos.add(x, y, z);

                    if (client.world.getBlockState(pos).getBlock() == Blocks.CHEST) {
                        if (excludeVisited && visitedChests.contains(pos)) continue;

                        double distance = playerPos.getSquaredDistance(pos);
                        if (distance < nearestDistance) {
                            nearestDistance = distance;
                            nearestChest = pos;
                        }
                    }
                }
            }
        }

        // If no unvisited chests, clear visited set and try again
        if (nearestChest == null && excludeVisited && !visitedChests.isEmpty()) {
            visitedChests.clear();
            return findNearestChest(client, false);
        }

        return nearestChest;
    }

    private BlockPos findNearestShulkerBox(MinecraftClient client) {
        BlockPos playerPos = client.player.getBlockPos();
        BlockPos nearestShulker = null;
        double nearestDistance = Double.MAX_VALUE;

        int range = 32;
        for (int x = -range; x <= range; x++) {
            for (int y = -range; y <= range; y++) {
                for (int z = -range; z <= range; z++) {
                    BlockPos pos = playerPos.add(x, y, z);

                    if (client.world.getBlockState(pos).getBlock().toString().contains("shulker")) {
                        double distance = playerPos.getSquaredDistance(pos);
                        if (distance < nearestDistance) {
                            nearestDistance = distance;
                            nearestShulker = pos;
                        }
                    }
                }
            }
        }

        return nearestShulker;
    }

    private void log(MinecraftClient client, String message) {
        client.execute(() -> {
            if (client.player != null) {
                client.player.sendMessage(Text.literal("§6[Chaos] " + message), false);
            }
        });
    }
}
