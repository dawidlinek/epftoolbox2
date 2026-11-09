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
                        { label: 'CLI Reference', link: '/getting-started/cli/' },
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
                        { label: 'Comparing Models', link: '/model-pipeline/comparing-models/' },
                        { label: 'Caching', link: '/model-pipeline/caching/' },
                        { label: 'Serialization', link: '/model-pipeline/serialization/' },
                        { label: 'Forecasting', link: '/model-pipeline/forecasting/' },
                        { label: 'Information Set', link: '/model-pipeline/information-set/' },
                        { label: 'Verifying a Pipeline', link: '/model-pipeline/verification/' },
                        { label: 'Time and Calendars', link: '/model-pipeline/time-and-calendars/' },
                        { label: 'Training Windows', link: '/model-pipeline/training-windows/' },
                        { label: 'Predictor Functions', link: '/model-pipeline/predictor-functions/' },
                        { label: 'Validation and Test', link: '/model-pipeline/validation-and-test/' },
                        { label: 'Ensembles', link: '/model-pipeline/ensembles/' },
                        { label: 'Hyper-parameter Search', link: '/model-pipeline/hyperparameter-search/' },
                        { label: 'Several Zones', link: '/model-pipeline/zones/' },
                        { label: 'Generation Targets', link: '/model-pipeline/generation/' },
                        { label: 'Chaining Pipelines', link: '/model-pipeline/chaining/' },
                        { label: 'The Engine', link: '/model-pipeline/engine/' },
                        { label: 'Predictor Presets', link: '/model-pipeline/presets/' },
                    ],
                },
                {
                    label: 'Workflow',
                    items: [
                        { label: 'Overview', link: '/workflow/overview/' },
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
                    label: 'Reproductions',
                    items: [
                        { label: 'Pooled Week-ahead Study', link: '/reproductions/pooled/' },
                    ],
                },
                {
                    label: 'Reference',
                    items: [
                        { label: 'API Reference', link: '/reference/api/' },
                        { label: 'Extending', link: '/reference/extending/' },
                        { label: 'Release Notes', link: '/reference/release-notes/' },
                    ],
                },
            ],
            customCss: ['./src/styles/custom.css'],
        }),
    ],
});
