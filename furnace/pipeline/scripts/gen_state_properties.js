#!/usr/bin/env node
/**
 * gen_state_properties.js
 * Generates state_properties_{version}.json: stateId -> {name, properties}.
 *
 * Usage (from furnace/pipeline/ dir where node_modules lives):
 *   node path/to/gen_state_properties.js <version> <output_path>
 */

const fs = require('fs');
const path = require('path');

const version = process.argv[2] || '1.19.4';
const outputPath = process.argv[3];

if (!outputPath) {
    console.error('Usage: node gen_state_properties.js <version> <output_path>');
    process.exit(1);
}

// minecraft-data lives in furnace/pipeline/node_modules — two levels up from here
const PIPELINE_DIR = path.join(__dirname, '..', '..', 'pipeline');
const mcData = require(path.join(PIPELINE_DIR, 'node_modules', 'minecraft-data'))(version);
const out = {};

for (const block of Object.values(mcData.blocks)) {
    const props = block.states || [];
    const total = props.length > 0
        ? props.reduce((a, p) => a * p.num_values, 1)
        : 1;

    for (let i = 0; i < total; i++) {
        let rem = i;
        const propVals = {};
        for (let j = props.length - 1; j >= 0; j--) {
            const p = props[j];
            const idx = rem % p.num_values;
            rem = Math.floor(rem / p.num_values);
            if (p.type === 'bool') {
                propVals[p.name] = idx === 0 ? 'true' : 'false';
            } else if (p.values) {
                propVals[p.name] = p.values[idx];
            } else {
                propVals[p.name] = String(idx);
            }
        }
        out[block.minStateId + i] = {
            name: 'minecraft:' + block.name,
            properties: propVals,
        };
    }
}

fs.writeFileSync(outputPath, JSON.stringify(out));
console.log('Wrote', Object.keys(out).length, 'state entries to', outputPath);
