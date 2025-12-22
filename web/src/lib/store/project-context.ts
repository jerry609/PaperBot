
import { create } from 'zustand'

export interface VirtualFile {
    name: string
    content: string
    language: string
}

interface ProjectState {
    files: Record<string, VirtualFile>
    activeFile: string | null

    // Actions
    addFile: (name: string, content: string, language?: string) => void
    updateFile: (name: string, content: string) => void
    setActiveFile: (name: string) => void
    removeFile: (name: string) => void
}

const DEFAULT_FILES: Record<string, VirtualFile> = {
    "model.py": {
        name: "model.py",
        language: "python",
        content: `import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output`
    },
    "train.py": {
        name: "train.py",
        language: "python",
        content: `import torch
from model import Transformer

def train():
    model = Transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # TODO: Implement training loop`
    },
    "config.yaml": {
        name: "config.yaml",
        language: "yaml",
        content: `model:
  d_model: 512
  nhead: 8
training:
  batch_size: 32
  epochs: 10`
    }
}

export const useProjectContext = create<ProjectState>((set) => ({
    files: DEFAULT_FILES,
    activeFile: "model.py",

    addFile: (name, content, language = "python") => set((state) => ({
        files: {
            ...state.files,
            [name]: { name, content, language }
        },
        activeFile: name // Switch to new file
    })),

    updateFile: (name, content) => set((state) => ({
        files: {
            ...state.files,
            [name]: { ...state.files[name], content }
        }
    })),

    setActiveFile: (name) => set({ activeFile: name }),

    removeFile: (name) => set((state) => {
        const newFiles = { ...state.files }
        delete newFiles[name]
        return {
            files: newFiles,
            activeFile: state.activeFile === name ? Object.keys(newFiles)[0] || null : state.activeFile
        }
    })
}))
