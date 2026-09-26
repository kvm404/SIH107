import {
  ArrowDown as LucideArrowDown,
  ArrowUp as LucideArrowUp,
  ArrowUpRight as LucideArrowUpRight,
  Check as LucideCheck,
  Clock as LucideClock,
  ChevronDown as LucideChevronDown,
  Copy as LucideCopy,
  Download as LucideDownload,
  ExternalLink as LucideExternalLink,
  FlaskConical as LucideFlask,
  Globe as LucideGlobe,
  Info as LucideInfo,
  LoaderCircle as LucideLoader,
  Mic as LucideMic,
  MicOff as LucideMicOff,
  Moon as LucideMoon,
  PanelLeft as LucidePanelLeft,
  Plus as LucidePlus,
  RotateCcw as LucideRetry,
  SquarePen as LucideNewChat,
  Sun as LucideSun,
  ThumbsDown as LucideThumbsDown,
  ThumbsUp as LucideThumbsUp,
  TriangleAlert as LucideAlert,
  X as LucideX,
} from "lucide-react";

interface IconProps {
  className?: string;
  size?: number;
}

function base(className: string | undefined, size: number | undefined) {
  return {
    className,
    size,
    "aria-hidden": true as const,
    style: { flexShrink: 0 },
  };
}

export function PlusIcon({ className, size = 16 }: IconProps) {
  return <LucidePlus {...base(className, size)} />;
}

export function NewChatIcon({ className, size = 16 }: IconProps) {
  return <LucideNewChat {...base(className, size)} />;
}

export function ArrowUpIcon({ className, size = 16 }: IconProps) {
  return <LucideArrowUp {...base(className, size)} />;
}

export function ArrowDownIcon({ className, size = 16 }: IconProps) {
  return <LucideArrowDown {...base(className, size)} />;
}

export function ArrowUpRightIcon({ className, size = 16 }: IconProps) {
  return <LucideArrowUpRight {...base(className, size)} />;
}

export function XIcon({ className, size = 18 }: IconProps) {
  return <LucideX {...base(className, size)} />;
}

export function SunIcon({ className, size = 20 }: IconProps) {
  return <LucideSun {...base(className, size)} />;
}

export function MoonIcon({ className, size = 20 }: IconProps) {
  return <LucideMoon {...base(className, size)} />;
}

export function MicIcon({ className, size = 18 }: IconProps) {
  return <LucideMic {...base(className, size)} />;
}

export function MicOffIcon({ className, size = 18 }: IconProps) {
  return <LucideMicOff {...base(className, size)} />;
}

export function SpinnerIcon({ className, size = 18 }: IconProps) {
  return <LucideLoader {...base(className, size)} />;
}

export function DownloadIcon({ className, size = 16 }: IconProps) {
  return <LucideDownload {...base(className, size)} />;
}

export function CheckIcon({ className, size = 14 }: IconProps) {
  return <LucideCheck {...base(className, size)} />;
}

export function CopyIcon({ className, size = 14 }: IconProps) {
  return <LucideCopy {...base(className, size)} />;
}

export function ExternalLinkIcon({ className, size = 14 }: IconProps) {
  return <LucideExternalLink {...base(className, size)} />;
}

export function ThumbsUpIcon({ className, size = 16 }: IconProps) {
  return <LucideThumbsUp {...base(className, size)} />;
}

export function ThumbsDownIcon({ className, size = 16 }: IconProps) {
  return <LucideThumbsDown {...base(className, size)} />;
}

export function AlertTriangleIcon({ className, size = 16 }: IconProps) {
  return <LucideAlert {...base(className, size)} />;
}

export function SidebarToggleIcon({ className, size = 20 }: IconProps) {
  return <LucidePanelLeft {...base(className, size)} />;
}

export function ChevronDownIcon({ className, size = 14 }: IconProps) {
  return <LucideChevronDown {...base(className, size)} />;
}

export function RetryIcon({ className, size = 14 }: IconProps) {
  return <LucideRetry {...base(className, size)} />;
}

export function DevIcon({ className, size = 16 }: IconProps) {
  return <LucideFlask {...base(className, size)} />;
}

export function ClockIcon({ className, size = 16 }: IconProps) {
  return <LucideClock {...base(className, size)} />;
}

export function InfoIcon({ className, size = 16 }: IconProps) {
  return <LucideInfo {...base(className, size)} />;
}

export function GlobeIcon({ className, size = 15 }: IconProps) {
  return <LucideGlobe {...base(className, size)} />;
}

/** Product brand mark (custom — the only non-library icon). */
export function ManakEmblemIcon({ className = "", size = 32 }: IconProps) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 36 36"
      fill="none"
      aria-hidden="true"
      style={{ flexShrink: 0 }}
    >
      <rect width="36" height="36" rx="9" fill="#065f46" />
      <path
        d="M10 26V12l5.2 8.4L20.4 12V26h-3.4V17.8l-1.8 3h-2.4l-1.8-3V26H10z"
        fill="#ffffff"
      />
      <circle cx="25.5" cy="11" r="2.6" fill="#6ee7b7" />
    </svg>
  );
}
