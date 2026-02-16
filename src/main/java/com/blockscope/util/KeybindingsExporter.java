package com.blockscope.util;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;

import java.util.LinkedHashMap;
import java.util.Map;

public class KeybindingsExporter {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    public static String exportKeybindings() {
        MinecraftClient client = MinecraftClient.getInstance();
        Map<String, KeybindInfo> keybindings = new LinkedHashMap<>();

        for (KeyBinding keybind : client.options.keysAll) {
            KeybindInfo info = new KeybindInfo();
            info.translationKey = keybind.getTranslationKey();
            info.category = keybind.getCategory();
            info.boundKey = keybind.getBoundKeyTranslationKey();
            info.defaultKey = keybind.getDefaultKey().getTranslationKey();
            info.isDefault = keybind.getBoundKeyTranslationKey().equals(keybind.getDefaultKey().getTranslationKey());

            keybindings.put(info.translationKey, info);
        }

        return GSON.toJson(keybindings);
    }

    public static class KeybindInfo {
        public String translationKey;  // e.g., "key.forward"
        public String category;         // e.g., "key.categories.movement"
        public String boundKey;         // Current bound key (e.g., "key.keyboard.w")
        public String defaultKey;       // Vanilla default key
        public boolean isDefault;       // True if user hasn't changed it
    }
}
