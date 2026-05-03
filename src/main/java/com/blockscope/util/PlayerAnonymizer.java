package com.blockscope.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * Anonymizes player names and UUIDs for privacy.
 * Maps real identifiers to random anonymous ones consistently within a session.
 */
public class PlayerAnonymizer {
    private static PlayerAnonymizer instance;

    private final Map<String, String> uuidMap = new HashMap<>();
    private final Map<String, String> nameMap = new HashMap<>();
    private final Random random = new Random();

    // Random name components for generating anonymous names
    private static final String[] ADJECTIVES = {
        "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Pink", "Gray",
        "Swift", "Bold", "Brave", "Quick", "Silent", "Wild", "Calm", "Fierce",
        "Bright", "Dark", "Light", "Shadow", "Storm", "Cloud", "Wind", "Fire"
    };

    private static final String[] NOUNS = {
        "Fox", "Wolf", "Bear", "Eagle", "Hawk", "Tiger", "Lion", "Panda",
        "Dragon", "Phoenix", "Falcon", "Raven", "Owl", "Snake", "Shark", "Whale",
        "Turtle", "Rabbit", "Deer", "Horse", "Cat", "Dog", "Bird", "Fish"
    };

    private PlayerAnonymizer() {}

    public static PlayerAnonymizer getInstance() {
        if (instance == null) {
            instance = new PlayerAnonymizer();
        }
        return instance;
    }

    /**
     * Anonymize a UUID string.
     * Returns a consistent anonymous UUID for the same input UUID within this session.
     */
    public String anonymizeUuid(String realUuid) {
        if (realUuid == null) return null;

        return uuidMap.computeIfAbsent(realUuid, k -> {
            // Generate a random but valid UUID
            return UUID.randomUUID().toString();
        });
    }

    /**
     * Anonymize a player/entity name.
     * Returns a consistent anonymous name for the same input name within this session.
     */
    public String anonymizeName(String realName) {
        if (realName == null) return null;

        return nameMap.computeIfAbsent(realName, k -> {
            // Generate random name like "RedFox" or "BraveEagle"
            String adjective = ADJECTIVES[random.nextInt(ADJECTIVES.length)];
            String noun = NOUNS[random.nextInt(NOUNS.length)];
            String baseName = adjective + noun;

            // Add number if collision
            String finalName = baseName;
            int suffix = 1;
            while (nameMap.containsValue(finalName)) {
                finalName = baseName + suffix++;
            }

            return finalName;
        });
    }

    /**
     * Clear all mappings (call at start of new recording session)
     * NOTE: Don't call this if you want names consistent across sessions!
     */
    public void resetSession() {
        // Commenting out - keep names consistent across sessions
        // uuidMap.clear();
        // nameMap.clear();
    }

    /**
     * Get number of unique players/entities anonymized this session
     */
    public int getAnonymizedCount() {
        return uuidMap.size();
    }
}
