import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMermaid from 'remark-mermaidjs';

export default defineConfig({
    site: 'https://dawidlinek.github.io',
    base: '/epftoolbox2',
    redirects: {
        '/': '/epftoolbox2/getting-started/introduction/',
    },
    markdown: {
        remarkPlugins: [
            [remarkMermaid, {
                mermaidConfig: {
                    theme: 'neutral',
                },
            }],
        ],
    },
    integrations: [
        starlight({
            title: 'epftoolbox2',
            description: 'A Python library for electricity price forecasting with modular data pipelines and model evaluation.',
            social: {
                github: 'https://github.com/dawidlinek/epftoolbox2',
            },
            sidebar: [
                {
                    label: 'Getting Started',
                    items: [
                        { label: 'Introduction', link: '/getting-started/introduction/' },
                        { label: 'Installation', link: '/getting-started/installation/' },
                        { label: 'Quick Start', link: '/getting-started/quickstart/' },
                    ],
                },
                {
                    label: 'Data Pipeline',
                    items: [
                        { label: 'Overview', link: '/data-pipeline/overview/' },
                        { label: 'Examples', link: '/data-pipeline/examples/' },
                        { label: 'Caching', link: '/data-pipeline/caching/' },
                        { label: 'Serialization', link: '/data-pipeline/serialization/' },
                    ],
                },
                {
                    label: 'Data Sources',
                    autogenerate: { directory: 'data-sources' },
                },
                {
                    label: 'Transformers',
                    autogenerate: { directory: 'transformers' },
                },
                {
                    label: 'Validators',
                    autogenerate: { directory: 'validators' },
                },
                {
                    label: 'Model Pipeline',
                    items: [
                        { label: 'Overview', link: '/model-pipeline/overview/' },
                        { label: 'Examples', link: '/model-pipeline/examples/' },
                        { label: 'Results', link: '/model-pipeline/results/' },
                        { label: 'Caching', link: '/model-pipeline/caching/' },
                    ],
                },
                {
                    label: 'Models',
                    autogenerate: { directory: 'models' },
                },
                {
                    label: 'Evaluators',
                    autogenerate: { directory: 'evaluators' },
                },
                {
                    label: 'Exporters',
                    autogenerate: { directory: 'exporters' },
                },
                {
                    label: 'Reference',
                    items: [
                        { label: 'API Reference', link: '/reference/api/' },
                        { label: 'Extending', link: '/reference/extending/' },
                    ],
                },
            ],
            customCss: ['./src/styles/custom.css'],
        }),
    ],
});
