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
  EllipsisVertical as LucideMore,
  PenLine as LucidePen,
  Trash2 as LucideTrash,
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
import brandMarkUrl from "./assets/bis-saarthi-mark.webp";

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

export function MoreIcon({ className, size = 16 }: IconProps) {
  return <LucideMore {...base(className, size)} />;
}

export function PenIcon({ className, size = 14 }: IconProps) {
  return <LucidePen {...base(className, size)} />;
}

export function TrashIcon({ className, size = 14 }: IconProps) {
  return <LucideTrash {...base(className, size)} />;
}

export function GlobeIcon({ className, size = 15 }: IconProps) {
  return <LucideGlobe {...base(className, size)} />;
}

/** BIS Saarthi brand mark: the leaves-and-sun symbol from the team's logo. */
export function BrandEmblemIcon({ className, size = 32 }: IconProps) {
  return (
    <img
      className={className}
      src={brandMarkUrl}
      width={size}
      height={size}
      alt=""
      aria-hidden="true"
      draggable={false}
      style={{ flexShrink: 0, display: "block" }}
    />
  );
}
