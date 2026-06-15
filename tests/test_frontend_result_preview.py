import subprocess


def test_result_preview_waits_for_image_load_before_showing():
    script = r"""
const fs = require('fs');
const vm = require('vm');

async function main() {
class MockElement {
  constructor({ hidden = false, files = [] } = {}) {
    this.hidden = hidden;
    this.files = files;
    this.textContent = '';
    this.className = '';
    this.src = '';
    this.listeners = new Map();
    this.classList = {
      add: (name) => {
        this.className = this.className ? `${this.className} ${name}` : name;
      },
    };
  }

  addEventListener(name, handler) {
    this.listeners.set(name, handler);
  }

  removeAttribute(name) {
    if (name === 'src') {
      this.src = '';
    }
  }

  async dispatch(name, payload = {}) {
    const handler = this.listeners.get(name);
    if (!handler) {
      return;
    }
    return await handler(payload);
  }
}

const form = new MockElement();
const fileInput = new MockElement({ files: [{ name: 'lane.png' }] });
const sourcePreview = new MockElement({ hidden: true });
const sourcePlaceholder = new MockElement();
const resultPreview = new MockElement({ hidden: true });
const resultPlaceholder = new MockElement();
const statusText = new MockElement();

const elements = {
  'predict-form': form,
  'image-input': fileInput,
  'source-preview': sourcePreview,
  'source-placeholder': sourcePlaceholder,
  'result-preview': resultPreview,
  'result-placeholder': resultPlaceholder,
  'form-status': statusText,
};

const context = {
  console,
  document: {
    getElementById(id) {
      return elements[id] ?? null;
    },
  },
  URL: {
    createObjectURL() {
      return 'blob:preview';
    },
  },
  FormData: class {
    append() {}
  },
  fetch: async () => ({
    ok: true,
    async json() {
      return { result_url: '/static/generated/missing.png' };
    },
  }),
};

vm.createContext(context);
vm.runInContext(fs.readFileSync('static/app.js', 'utf8'), context);

const submitPromise = form.dispatch('submit', {
  preventDefault() {},
});
await Promise.resolve();

if (resultPreview.hidden !== true) {
  throw new Error('result preview should stay hidden until the image loads');
}

if (resultPlaceholder.hidden !== false) {
  throw new Error('result placeholder should stay visible until the image loads');
}

void submitPromise;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
"""

    result = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
