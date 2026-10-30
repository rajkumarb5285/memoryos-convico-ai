import { useState, useEffect } from 'react';
import {
  Brain,
  Sparkles,
  Send,
  Database,
  Layers,
  Search,
  Trash2,
  User,
  Activity,
  RefreshCw,
  Clock,
  Key,
} from 'lucide-react';

interface ExtractedMemory {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  memory_type: string;
  domain: string;
  polarity: string;
  temporal_status: string;
  valid_from: string;
  valid_until: string | null;
  confidence: number;
}

interface RetrievalEvidence {
  memory_id: string;
  fact: string;
  subject: string;
  predicate: string;
  object: string;
  memory_type: string;
  domain: string;
  temporal_status: string;
  confidence: number;
  relevance_score: number;
  retrieval_lanes: string[];
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  factsExtracted?: string[];
}

export default function App() {
  // State
  const [activeTab, setActiveTab] = useState<'memories' | 'retrieval'>('memories');
  const [token, setToken] = useState<string>('');
  const [userId, setUserId] = useState<string>('');
  const [characterId, setCharacterId] = useState<string>('');
  const [conversationId, setConversationId] = useState<string>('');
  
  const [chatInput, setChatInput] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content: 'Hello! I am MEMORYOS, your intelligent conversational memory engine. Talk to me in English or Hinglish—I automatically extract, store, resolve contradictions, and recall your personal context across conversations.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  
  const [memories, setMemories] = useState<ExtractedMemory[]>([]);
  const [loadingIngest, setLoadingIngest] = useState<boolean>(false);
  const [loadingMemories, setLoadingMemories] = useState<boolean>(false);
  
  // Retrieval Explorer State
  const [retrievalQuery, setRetrievalQuery] = useState<string>('Where do I live currently?');
  const [retrievalEvidence, setRetrievalEvidence] = useState<RetrievalEvidence[]>([]);
  const [retrievalConfidence, setRetrievalConfidence] = useState<number | null>(null);
  const [loadingRetrieval, setLoadingRetrieval] = useState<boolean>(false);
  
  // Health
  const [healthStatus, setHealthStatus] = useState<{ postgres: boolean; redis: boolean }>({
    postgres: true,
    redis: true,
  });

  // Initialize or fetch auth & character on mount
  useEffect(() => {
    checkHealth();
    const savedToken = localStorage.getItem('memoryos_token');
    const savedUid = localStorage.getItem('memoryos_uid');
    const savedCid = localStorage.getItem('memoryos_cid');

    if (savedToken && savedUid && savedCid) {
      setToken(savedToken);
      setUserId(savedUid);
      setCharacterId(savedCid);
      fetchMemories(savedUid, savedCid, savedToken);
    } else {
      bootstrapSession();
    }
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        const dbOk = data.checks?.database === 'healthy' || data.checks?.database === 'connected';
        const redisOk = data.checks?.cache === 'healthy' || data.checks?.cache === 'connected';
        setHealthStatus({
          postgres: Boolean(dbOk),
          redis: Boolean(redisOk),
        });
      }
    } catch {
      setHealthStatus({ postgres: false, redis: false });
    }
  };

  const bootstrapSession = async () => {
    try {
      const username = `apple_user_${Math.floor(Math.random() * 100000)}`;
      const password = 'Password123!';
      const email = `${username}@memoryos.apple.ai`;

      // 1. Register User
      const regRes = await fetch('/api/v1/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });

      if (!regRes.ok) return null;

      // 2. Obtain JWT Access Token
      const tokenRes = await fetch('/api/v1/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });

      if (!tokenRes.ok) return null;
      const tokenData = await tokenRes.json();
      const userToken = tokenData.access_token;
      const uid = tokenData.user_id;

      setToken(userToken);
      setUserId(uid);
      localStorage.setItem('memoryos_token', userToken);
      localStorage.setItem('memoryos_uid', uid);

      // 3. Create Default Character Scope
      const charRes = await fetch('/api/v1/character', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${userToken}`,
        },
        body: JSON.stringify({
          name: 'Jarvis (Personal Assistant)',
          description: 'Default memory scope for user personal assertions',
        }),
      });

      if (charRes.ok) {
        const charData = await charRes.json();
        setCharacterId(charData.id);
        localStorage.setItem('memoryos_cid', charData.id);
        fetchMemories(uid, charData.id, userToken);
        return { token: userToken, userId: uid, characterId: charData.id };
      }
    } catch (err) {
      console.error('Session bootstrap error:', err);
    }
    return null;
  };

  const fetchMemories = async (uid: string, cid: string, authToken: string) => {
    if (!uid || !authToken) return;
    setLoadingMemories(true);
    try {
      const res = await fetch(`/api/v1/memory/${uid}?character_id=${cid}&limit=50`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        setMemories(data.memories || []);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingMemories(false);
    }
  };

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || chatInput;
    if (!text.trim()) return;

    let activeToken = token;
    let activeCharId = characterId;
    let activeUid = userId;

    if (!activeToken || !activeCharId) {
      const session = await bootstrapSession();
      if (!session) {
        alert('Initializing session with MEMORYOS backend... Please try again in 2 seconds.');
        return;
      }
      activeToken = session.token;
      activeCharId = session.characterId;
      activeUid = session.userId;
    }

    const newMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newMsg]);
    setChatInput('');
    setLoadingIngest(true);

    try {
      const res = await fetch('/api/v1/memory/ingest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${activeToken}`,
        },
        body: JSON.stringify({
          character_id: activeCharId,
          conversation_id: conversationId || undefined,
          messages: [{ role: 'user', content: text }],
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.conversation_id && !conversationId) {
          setConversationId(data.conversation_id);
        }

        const factCount = data.extracted_facts_count || 0;
        let assistantReply = `Understood. I processed your message. (${factCount} memories extracted & committed to PostgreSQL).`;

        if (factCount === 0) {
          assistantReply = 'I listened to your input. No new personal assertions or facts were detected to store.';
        }

        setMessages((prev) => [
          ...prev,
          {
            id: `reply-${Date.now()}`,
            role: 'assistant',
            content: assistantReply,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);

        // Refresh memories
        fetchMemories(activeUid, activeCharId, activeToken);
      } else {
        const errJson = await res.json().catch(() => ({}));
        setMessages((prev) => [
          ...prev,
          {
            id: `err-${Date.now()}`,
            role: 'assistant',
            content: `Error from server: ${errJson.message || res.statusText || 'Failed to process message'}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
      }
    } catch (err: unknown) {
      console.error('Ingest error:', err);
      const errMsg = err instanceof Error ? err.message : 'Unable to connect to backend';
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          role: 'assistant',
          content: `Network connection error: ${errMsg}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setLoadingIngest(false);
    }
  };

  const handleExecuteRetrieval = async () => {
    if (!retrievalQuery.trim()) return;

    let activeToken = token;
    let activeCharId = characterId;

    if (!activeToken || !activeCharId) {
      const session = await bootstrapSession();
      if (!session) return;
      activeToken = session.token;
      activeCharId = session.characterId;
    }

    setLoadingRetrieval(true);
    try {
      const res = await fetch('/api/v1/memory/retrieve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${activeToken}`,
        },
        body: JSON.stringify({
          query: retrievalQuery,
          character_id: activeCharId,
          limit: 5,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setRetrievalEvidence(data.evidence || []);
        setRetrievalConfidence(data.confidence_score);
      }
    } catch (err) {
      console.error('Retrieval error:', err);
    } finally {
      setLoadingRetrieval(false);
    }
  };

  const handleDeleteMemory = async (memoryId: string) => {
    if (!token) return;
    try {
      const res = await fetch(`/api/v1/memory/${memoryId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        setMemories((prev) => prev.filter((m) => m.id !== memoryId));
      }
    } catch (e) {
      console.error('Delete error:', e);
    }
  };

  return (
    <div className="flex flex-col h-screen select-none">
      {/* 1. Apple Navigation Bar & Dynamic Island */}
      <header className="h-16 px-6 flex items-center justify-between border-b border-white/10 glass-panel-subtle z-20">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl glass-button flex items-center justify-center text-blue-400 shadow-lg">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight text-white">MEMORYOS</span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-400/30">
                PROD v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 font-normal">Intelligent Conversational Memory Engine</p>
          </div>
        </div>

        {/* Dynamic Island Status Pill */}
        <div className="dynamic-island px-4 py-1.5 rounded-full flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${healthStatus.postgres ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-slate-300">PostgreSQL (5438)</span>
          </div>
          <span className="text-slate-600">|</span>
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${healthStatus.redis ? 'bg-emerald-400 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-slate-300">Redis (6385)</span>
          </div>
          <span className="text-slate-600">|</span>
          <div className="flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-blue-300 font-medium">Groq / Neural Hybrid</span>
          </div>
        </div>

        {/* User Identity Pill */}
        <div className="flex items-center gap-3">
          <div className="glass-panel px-3 py-1.5 rounded-full flex items-center gap-2 text-xs text-slate-300">
            <User className="w-3.5 h-3.5 text-blue-400" />
            <span>IDOR-Safe User Isolation</span>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex flex-1 overflow-hidden p-4 gap-4">
        {/* Left / Center: Interactive Dialogue Canvas */}
        <div className="flex-1 flex flex-col glass-panel rounded-3xl overflow-hidden shadow-2xl">
          {/* Conversation Header */}
          <div className="px-6 py-4 border-b border-white/10 flex items-center justify-between bg-white/[0.02]">
            <div className="flex items-center gap-2.5">
              <Activity className="w-4 h-4 text-blue-400" />
              <h2 className="text-sm font-semibold tracking-tight text-white">Live Multi-Turn Dialogue</h2>
              <span className="text-xs text-slate-400 font-normal ml-2">Character Scope: Jarvis</span>
            </div>
            <button
              onClick={() => fetchMemories(userId, characterId, token)}
              className="glass-button p-2 rounded-xl text-slate-300 hover:text-white"
              title="Refresh memories"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>

          {/* Quick Test Prompt Pills */}
          <div className="px-6 py-3 border-b border-white/5 bg-black/20 flex items-center gap-2 overflow-x-auto text-xs">
            <span className="text-slate-400 whitespace-nowrap flex items-center gap-1 text-[11px] uppercase tracking-wider font-semibold">
              <Key className="w-3 h-3 text-blue-400" /> Quick Tests:
            </span>
            <button
              onClick={() => handleSendMessage('I work as a senior software engineer at Google')}
              className="glass-button px-3 py-1 rounded-full text-slate-200 whitespace-nowrap hover:border-blue-400/50"
            >
              Career Fact
            </button>
            <button
              onClick={() => handleSendMessage('mujhe chai bahut pasand hai lekin coffee pasand nahi hai')}
              className="glass-button px-3 py-1 rounded-full text-amber-200 whitespace-nowrap hover:border-amber-400/50"
            >
              Hinglish Negation
            </button>
            <button
              onClick={() => handleSendMessage('I lived in Indore before moving to Pune')}
              className="glass-button px-3 py-1 rounded-full text-emerald-200 whitespace-nowrap hover:border-emerald-400/50"
            >
              Temporal Supersession
            </button>
            <button
              onClick={() => handleSendMessage('Do you know someone at Microsoft?')}
              className="glass-button px-3 py-1 rounded-full text-rose-200 whitespace-nowrap hover:border-rose-400/50"
            >
              Question (Noise Filter)
            </button>
          </div>

          {/* Message Stream */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {m.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-xl glass-button flex items-center justify-center text-blue-400 shrink-0">
                    <Brain className="w-4 h-4" />
                  </div>
                )}
                <div
                  className={`max-w-[75%] rounded-2xl p-4 text-sm leading-relaxed ${
                    m.role === 'user'
                      ? 'glass-button-primary text-white'
                      : 'glass-panel text-slate-200 border-white/10'
                  }`}
                >
                  <p>{m.content}</p>
                  <span className="text-[10px] text-white/50 block mt-2 text-right">{m.timestamp}</span>
                </div>
                {m.role === 'user' && (
                  <div className="w-8 h-8 rounded-xl bg-blue-600 flex items-center justify-center text-white shrink-0 shadow-lg">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            ))}
            {loadingIngest && (
              <div className="flex items-center gap-2 text-xs text-blue-400 animate-pulse pl-11">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Running neural memory extraction & contradiction resolver...</span>
              </div>
            )}
          </div>

          {/* Glass Input Bar */}
          <div className="p-4 border-t border-white/10 bg-white/[0.01]">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex items-center gap-3"
            >
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Tell MEMORYOS about your life, career, or preferences in English or Hinglish..."
                className="flex-1 glass-input px-5 py-3.5 rounded-2xl text-sm"
              />
              <button
                type="submit"
                disabled={loadingIngest || !chatInput.trim()}
                className="glass-button-primary px-6 py-3.5 rounded-2xl flex items-center gap-2 text-sm font-medium disabled:opacity-50 text-white"
              >
                <span>Send</span>
                <Send className="w-4 h-4" />
              </button>
            </form>
          </div>
        </div>

        {/* Right Panel: Apple Live Inspector (Memories & Hybrid RAG) */}
        <div className="w-[440px] flex flex-col glass-panel rounded-3xl overflow-hidden shadow-2xl">
          {/* Segmented Control Header */}
          <div className="p-3 border-b border-white/10 bg-black/20 flex gap-2">
            <button
              onClick={() => setActiveTab('memories')}
              className={`flex-1 py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                activeTab === 'memories'
                  ? 'glass-button text-white shadow-lg border-white/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Database className="w-3.5 h-3.5 text-blue-400" />
              <span>Durable Memories ({memories.length})</span>
            </button>
            <button
              onClick={() => setActiveTab('retrieval')}
              className={`flex-1 py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 transition-all ${
                activeTab === 'retrieval'
                  ? 'glass-button text-white shadow-lg border-white/30'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Hybrid RAG Lanes</span>
            </button>
          </div>

          {/* Tab 1: Extracted Memory Cards */}
          {activeTab === 'memories' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {loadingMemories ? (
                <div className="text-center py-12 text-slate-400 text-xs">Loading persistent memories...</div>
              ) : memories.length === 0 ? (
                <div className="text-center py-12 text-slate-400 text-xs flex flex-col items-center gap-3">
                  <Database className="w-8 h-8 text-slate-600" />
                  <span>No memories extracted yet. Speak to MEMORYOS to assert facts!</span>
                </div>
              ) : (
                memories.map((m) => (
                  <div
                    key={m.id}
                    className="glass-panel p-4 rounded-2xl border-white/10 hover:border-white/20 transition-all group relative"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                            m.temporal_status === 'current'
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-400/30'
                              : 'bg-amber-500/20 text-amber-300 border border-amber-400/30'
                          }`}
                        >
                          {m.temporal_status}
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          {m.domain}
                        </span>
                      </div>
                      <button
                        onClick={() => handleDeleteMemory(m.id)}
                        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-rose-400 transition-opacity p-1"
                        title="Delete memory"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <div className="text-sm font-medium text-white mb-1">
                      <span className="text-blue-400">{m.predicate}</span>: {m.object}
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-slate-400 mt-3 pt-2 border-t border-white/5">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-slate-500" />
                        {new Date(m.valid_from).toLocaleDateString()}
                      </span>
                      <span
                        className={`font-semibold ${
                          m.polarity === 'likes' || m.polarity === 'positive'
                            ? 'text-emerald-400'
                            : m.polarity === 'dislikes' || m.polarity === 'negative'
                            ? 'text-rose-400'
                            : 'text-slate-300'
                        }`}
                      >
                        {m.polarity}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 2: 3-Lane Hybrid RAG Visualizer */}
          {activeTab === 'retrieval' && (
            <div className="flex-1 flex flex-col overflow-hidden p-4 space-y-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <Search className="w-3.5 h-3.5 text-blue-400" /> Natural Language Query
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={retrievalQuery}
                    onChange={(e) => setRetrievalQuery(e.target.value)}
                    placeholder="Ask a question..."
                    className="flex-1 glass-input px-3.5 py-2 rounded-xl text-xs"
                  />
                  <button
                    onClick={handleExecuteRetrieval}
                    disabled={loadingRetrieval}
                    className="glass-button-primary px-4 py-2 rounded-xl text-xs font-semibold text-white flex items-center gap-1"
                  >
                    <span>Query</span>
                  </button>
                </div>
              </div>

              {/* 3 Lanes Breakdown Info */}
              <div className="grid grid-cols-3 gap-2 text-center text-[10px]">
                <div className="glass-panel p-2 rounded-xl border-blue-500/20">
                  <div className="text-blue-400 font-bold">Lane A</div>
                  <div className="text-slate-400">Structured</div>
                </div>
                <div className="glass-panel p-2 rounded-xl border-emerald-500/20">
                  <div className="text-emerald-400 font-bold">Lane B</div>
                  <div className="text-slate-400">Lexical / FTS</div>
                </div>
                <div className="glass-panel p-2 rounded-xl border-purple-500/20">
                  <div className="text-purple-400 font-bold">Lane C</div>
                  <div className="text-slate-400">pgvector Cosine</div>
                </div>
              </div>

              {/* RRF Confidence Meter */}
              {retrievalConfidence !== null && (
                <div className="glass-panel p-3 rounded-2xl flex items-center justify-between text-xs">
                  <span className="text-slate-300">RRF Reranker Confidence:</span>
                  <span className="font-bold text-emerald-400 text-sm">
                    {(retrievalConfidence * 100).toFixed(1)}%
                  </span>
                </div>
              )}

              {/* Retrieved Evidence Cards */}
              <div className="flex-1 overflow-y-auto space-y-2.5">
                {loadingRetrieval ? (
                  <div className="text-center py-10 text-xs text-slate-400">Executing 3-Lane RAG Fusion...</div>
                ) : retrievalEvidence.length === 0 ? (
                  <div className="text-center py-10 text-xs text-slate-400">
                    Run a query to inspect matched evidence and lane contributions.
                  </div>
                ) : (
                  retrievalEvidence.map((ev, idx) => (
                    <div key={idx} className="glass-panel p-3.5 rounded-2xl border-white/10 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-white">{ev.fact}</span>
                        <span className="text-[10px] text-blue-400 font-bold">
                          #{(ev.relevance_score).toFixed(2)}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 flex-wrap">
                        {ev.retrieval_lanes.map((lane) => (
                          <span
                            key={lane}
                            className="text-[9px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-300 border border-blue-400/20"
                          >
                            {lane}
                          </span>
                        ))}
                        <span className="text-[9px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-slate-500/10 text-slate-300">
                          {ev.temporal_status}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
