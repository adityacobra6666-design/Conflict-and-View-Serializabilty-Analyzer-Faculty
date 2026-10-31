import { PyqQuestion, PYQ_QUESTIONS } from '../data/pyqQuestions';

const STORAGE_KEY = 'pyq_exams_progress_v1';

export interface PyqProgress {
  attempts: Record<string, {
    questionId: string;
    selectedIndex: number;
    isCorrect: boolean;
    timestamp: string;
  }>;
  bookmarks: string[];
}

export const pyqService = {
  getAllQuestions(): PyqQuestion[] {
    return PYQ_QUESTIONS;
  },

  getExams(): string[] {
    const questions = this.getAllQuestions();
    const set = new Set<string>();
    questions.forEach(q => set.add(q.exam));
    return Array.from(set);
  },

  getYears(): number[] {
    const questions = this.getAllQuestions();
    const set = new Set<number>();
    questions.forEach(q => set.add(q.year));
    return Array.from(set).sort((a, b) => b - a);
  },

  getTopics(): string[] {
    const questions = this.getAllQuestions();
    const set = new Set<string>();
    questions.forEach(q => set.add(q.topic));
    return Array.from(set);
  },

  getProgress(): PyqProgress {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (raw) {
          return JSON.parse(raw);
        }
      }
    } catch (e) {
      // Safe fallback
    }
    return { attempts: {}, bookmarks: [] };
  },

  saveAttempt(questionId: string, selectedIndex: number, isCorrect: boolean): PyqProgress {
    const progress = this.getProgress();
    progress.attempts[questionId] = {
      questionId,
      selectedIndex,
      isCorrect,
      timestamp: new Date().toISOString()
    };
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
      }
    } catch (e) {
      // Safe fallback
    }
    return progress;
  },

  toggleBookmark(questionId: string): PyqProgress {
    const progress = this.getProgress();
    const idx = progress.bookmarks.indexOf(questionId);
    if (idx >= 0) {
      progress.bookmarks.splice(idx, 1);
    } else {
      progress.bookmarks.push(questionId);
    }
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
      }
    } catch (e) {
      // Safe fallback
    }
    return progress;
  },

  resetProgress(): PyqProgress {
    const initial: PyqProgress = { attempts: {}, bookmarks: [] };
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(initial));
      }
    } catch (e) {
      // Safe fallback
    }
    return initial;
  }
};
