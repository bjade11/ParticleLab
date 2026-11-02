"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";
import { FolderOpen, ImagePlus } from "lucide-react";

export const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"];

export function isImageFile(f: File): boolean {
  const n = f.name.toLowerCase();
  return IMAGE_EXTS.some((e) => n.endsWith(e));
}

export interface DropzoneProps {
  multiple?: boolean;
  /** 폴더 선택 (webkitdirectory). multiple 암시 */
  directory?: boolean;
  accept?: string;
  disabled?: boolean;
  onFiles: (files: File[]) => void;
  /** 비어 있을 때 표시 */
  children?: ReactNode;
  /** 채워진 상태 표현 */
  filled?: boolean;
  className?: string;
  compact?: boolean;
}

/** 드래그된 DataTransfer 에서 파일 수집 (디렉터리 드롭 시 재귀 탐색) */
async function collectFromDataTransfer(dt: DataTransfer): Promise<File[]> {
  const items = Array.from(dt.items ?? []);
  const entries = items
    .map((it) => (typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null))
    .filter((e): e is FileSystemEntry => !!e);
  if (entries.length === 0 || !entries.some((e) => e.isDirectory)) {
    return Array.from(dt.files ?? []);
  }
  const out: File[] = [];
  const readEntry = (entry: FileSystemEntry): Promise<void> =>
    new Promise((resolve) => {
      if (entry.isFile) {
        (entry as FileSystemFileEntry).file(
          (f) => {
            out.push(f);
            resolve();
          },
          () => resolve(),
        );
      } else if (entry.isDirectory) {
        const reader = (entry as FileSystemDirectoryEntry).createReader();
        const batch = () =>
          reader.readEntries(
            async (ents) => {
              if (ents.length === 0) {
                resolve();
                return;
              }
              await Promise.all(ents.map(readEntry));
              batch();
            },
            () => resolve(),
          );
        batch();
      } else resolve();
    });
  await Promise.all(entries.map(readEntry));
  return out;
}

export default function Dropzone({
  multiple,
  directory,
  accept,
  disabled,
  onFiles,
  children,
  filled,
  className,
  compact,
}: DropzoneProps) {
  const id = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [active, setActive] = useState(false);

  // webkitdirectory 는 React 타입에 없어 ref 로 설정
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    if (directory) {
      el.setAttribute("webkitdirectory", "");
      el.setAttribute("directory", "");
    } else {
      el.removeAttribute("webkitdirectory");
      el.removeAttribute("directory");
    }
  }, [directory]);

  const emit = (files: File[]) => {
    if (files.length === 0) return;
    onFiles(multiple || directory ? files : [files[0]]);
  };

  return (
    <div
      className={clsx(
        "dropzone relative flex flex-col items-center justify-center text-center text-text-mute",
        compact ? "px-3 py-3" : "px-4 py-6",
        active && "dropzone-active",
        filled && "dropzone-filled",
        disabled && "opacity-50 pointer-events-none",
        className,
      )}
      onDragOver={(e) => {
        e.preventDefault();
        if (!active) setActive(true);
      }}
      onDragLeave={() => setActive(false)}
      onDrop={async (e) => {
        e.preventDefault();
        setActive(false);
        const files = await collectFromDataTransfer(e.dataTransfer);
        emit(files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
      }}
    >
      <input
        id={id}
        ref={inputRef}
        type="file"
        className="hidden"
        multiple={multiple || directory}
        accept={accept}
        disabled={disabled}
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          emit(files);
          e.target.value = "";
        }}
      />
      {children ?? (
        <>
          {directory ? <FolderOpen size={20} className="mb-1.5 text-text-faint" /> : <ImagePlus size={20} className="mb-1.5 text-text-faint" />}
          <div className="text-[12px] font-medium">
            {directory ? "폴더를 끌어다 놓거나 클릭하여 선택" : multiple ? "이미지를 끌어다 놓거나 클릭하여 선택" : "이미지 1장을 끌어다 놓거나 클릭"}
          </div>
          <div className="text-[10px] text-text-faint mt-0.5">jpg · png · tif · bmp</div>
        </>
      )}
    </div>
  );
}
