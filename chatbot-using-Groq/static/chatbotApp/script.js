$(document).ready(function () {
  const chatWindow = $("#message-list");
  const fileInput = $("#file-input");
  const messageInput = $("#message-input");
  const sendButton = $("#send-button");
  const sessionList = $("#session-list");
  const paperClipBtn = $("#paper-clip-btn");
  const filePreview = document.getElementById('file-preview');
  const selectedFiles = new Set();

  // Initialize dark mode from localStorage
  const isDarkMode = localStorage.getItem('darkMode') === 'true';
  $("body").toggleClass("dark-mode", isDarkMode);
  $("#light-dark-mode-switch").prop('checked', isDarkMode);

  // Toggle dark mode
  $("#light-dark-mode-switch").on("change", function () {
    const isDark = this.checked;
    $("body").toggleClass("dark-mode", isDark);
    localStorage.setItem('darkMode', isDark);
  });

  let reader = null;
  let cancelRequested = false;
  let scrollLocked = false;

  // Session management
  let sessions = [];
  let currentSessionId = null;
  let sessionLock = {
    isLocked: false,
    lockTimeout: null,
    lock() {
      if (this.isLocked) return false;
      this.isLocked = true;
      // Clear any existing timeout
      if (this.lockTimeout) {
        clearTimeout(this.lockTimeout);
      }
      // Set timeout to automatically unlock after 5 seconds
      this.lockTimeout = setTimeout(() => {
        this.unlock();
      }, 5000);
      return true;
    },
    unlock() {
      this.isLocked = false;
      if (this.lockTimeout) {
        clearTimeout(this.lockTimeout);
        this.lockTimeout = null;
      }
    }
  };

  // Add cache for session data with server compatibility
  const sessionCache = {
    data: new Map(),
    lastUpdate: new Map(),
    maxAge: 30000, // 30 seconds
    maxRetries: 3,
    retryDelay: 1000, // 1 second

    get(sessionId) {
      const lastUpdate = this.lastUpdate.get(sessionId);
      if (lastUpdate && Date.now() - lastUpdate < this.maxAge) {
        return this.data.get(sessionId);
      }
      return null;
    },

    set(sessionId, data) {
      this.data.set(sessionId, data);
      this.lastUpdate.set(sessionId, Date.now());
    },

    clear(sessionId) {
      this.data.delete(sessionId);
      this.lastUpdate.delete(sessionId);
    }
  };

  // Add debounce function at the top level
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Function to fetch sessions from database
  function fetchSessionsFromDB() {
    return new Promise((resolve, reject) => {
      $.ajax({
        url: '/api/get_detailed_chat_history/',
        method: 'GET',
        success: function(response) {
          if (response.sidebar_history) {
            // Convert sidebar history to sessions format and remove duplicates
            const uniqueSessions = new Map();
            
            response.sidebar_history.forEach(conv => {
              // Skip if we already have this session
              if (uniqueSessions.has(conv.session_id)) {
                return;
              }
              
              // Get the first user message for this session
              const firstUserMessage = conv.user_message || "New Chat";
              
              // Store the session with its title
              uniqueSessions.set(conv.session_id, {
                id: conv.session_id,
                name: firstUserMessage, // Use the actual message as title
                timestamp: conv.timestamp
              });
            });
            
            // Convert Map to array and sort by timestamp
            sessions = Array.from(uniqueSessions.values()).sort((a, b) => {
              return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
            });
            
            updateSessionList();
          }
          resolve(response);
        },
        error: function(xhr, status, error) {
          console.error('Error fetching sessions:', error);
          reject(error);
        }
      });
    });
  }

  // Function to create new session
  function createNewSession() {
    // Try to acquire the lock
    if (!sessionLock.lock()) {
      console.log('Session creation is locked');
      return Promise.reject(new Error('Session creation is locked'));
    }

    return new Promise((resolve, reject) => {
      $.ajax({
        url: '/api/create_new_session/',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCSRFToken()
        },
        success: function(response) {
          if (response.status === 'success') {
            const newSession = {
              id: response.session_id,
              name: "New Chat",
              timestamp: response.timestamp
            };
            // Remove any existing sessions with the same ID
            sessions = sessions.filter(s => s.id !== response.session_id);
            sessions.unshift(newSession);
            updateSessionList();
            resolve(response.session_id);
          } else {
            reject(new Error('Failed to create session'));
          }
        },
        error: function(xhr, status, error) {
          console.error('Error creating session:', error);
          reject(error);
        },
        complete: function() {
          // Unlock after a short delay
          setTimeout(() => {
            sessionLock.unlock();
          }, 1000);
        }
      });
    });
  }

  // Function to update session name based on first user message
  function updateSessionName(sessionId, query) {
    const session = sessions.find(s => s.id === sessionId);
    if (session && (!session.name || session.name === "New Chat")) {
        // Clean and format the query for the title
        let title = query.trim();
        
        // Remove any special characters and limit length
        title = title.replace(/[^\w\s]/g, '');
        
        // If title is too long, truncate it and add ellipsis
        if (title.length > 30) {
            title = title.substring(0, 30) + "...";
        }
        
        // If title is empty after cleaning, use default
        if (!title) {
            title = "New Chat";
        }
        
        // Update the session name
        session.name = title;
        
        // Update the session name in the database
        $.ajax({
            url: '/api/update_session_name/',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            data: JSON.stringify({
                session_id: sessionId,
                name: title
            }),
            success: function(response) {
                console.log('Session name updated:', response);
                // Store the title in localStorage for persistence
                localStorage.setItem(`chat_title_${sessionId}`, title);
                updateSessionList();
            },
            error: function(xhr, status, error) {
                console.error('Error updating session name:', error);
            }
        });
    }
  }

  // Function to generate a meaningful session title
  function generateSessionTitle(query) {
    // Clean the query
    let title = query.trim();
    
    // Remove special characters and extra spaces
    title = title.replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
    
    // If the query is a question, remove the question mark and common question words
    title = title.replace(/^(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)\s+/i, '');
    
    // Capitalize first letter of each word
    title = title.split(' ').map(word => {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    }).join(' ');
    
    // If title is too long, truncate it and add ellipsis
    if (title.length > 30) {
        // Try to find a good breaking point
        const lastSpace = title.substring(0, 30).lastIndexOf(' ');
        if (lastSpace > 20) { // If we can find a good breaking point
            title = title.substring(0, lastSpace) + "...";
        } else {
            title = title.substring(0, 30) + "...";
        }
    }
    
    // If title is empty after cleaning, use default
    if (!title) {
        title = "New Chat";
    }
    
    return title;
  }

  // Function to update session list UI
  function updateSessionList() {
    sessionList.empty();
    
    // Sort sessions by timestamp (newest first)
    const sortedSessions = [...sessions].sort((a, b) => {
        return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
    });
    
    // Track unique sessions to prevent duplicates
    const uniqueSessions = new Set();
    
    sortedSessions.forEach(session => {
        // Skip if we've already processed this session
        if (uniqueSessions.has(session.id)) {
            return;
        }
        uniqueSessions.add(session.id);
        
        const isActive = session.id === currentSessionId;
        // Try to get the title from localStorage first, then fall back to session name
        const storedTitle = localStorage.getItem(`chat_title_${session.id}`);
        const sessionName = storedTitle || session.name || "New Chat";
        const timestamp = formatTimestamp(session.timestamp);
        
        const sessionHtml = `
            <div class="session-item ${isActive ? 'active' : ''}" data-session-id="${session.id}">
                <div class="session-preview" title="${sessionName}">
                    <div class="session-title">${sessionName}</div>
                    <div class="session-timestamp">${timestamp}</div>
                </div>
                <div class="session-actions">
                    <button class="rename-session" data-session-id="${session.id}" title="Rename chat">
                        <i class="fa fa-pencil"></i>
                    </button>
                    <button class="delete-session" data-session-id="${session.id}" title="Delete chat">
                        <i class="fa fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        sessionList.append(sessionHtml);
    });
  }

  // Function to delete a session
  function deleteSession(sessionId) {
    if (!confirm("Are you sure you want to delete this chat? This action cannot be undone.")) {
        return;
    }

    $.ajax({
        url: '/api/clear_chat_history/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        data: JSON.stringify({ session_id: sessionId }),
        success: function(response) {
            // Remove the session from the sessions array
            sessions = sessions.filter(s => s.id !== sessionId);
            
            // Remove the title from localStorage
            localStorage.removeItem(`chat_title_${sessionId}`);
            
            // If the deleted session was the current one, create a new session
            if (sessionId === currentSessionId) {
                createNewSession().then(newSessionId => {
                    switchSession(newSessionId);
                });
            }
            
            // Update the session list
            updateSessionList();
        },
        error: function(xhr, status, error) {
            console.error('Error deleting session:', error);
            alert('Failed to delete chat. Please try again.');
        }
    });
  }

  // Function to rename a session
  function renameSession(sessionId) {
    const session = sessions.find(s => s.id === sessionId);
    if (!session) return;

    const currentName = session.name || "New Chat";
    
    // Create and show the rename modal
    const modalHtml = `
        <div class="rename-modal-overlay">
            <div class="rename-modal">
                <div class="rename-modal-header">
                    <h3>Rename Chat</h3>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="rename-modal-body">
                    <input type="text" id="new-chat-name" class="rename-input" value="${currentName}" maxlength="50">
                    <div class="rename-modal-footer">
                        <button class="cancel-rename">Cancel</button>
                        <button class="save-rename">Save</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to body
    $('body').append(modalHtml);
    
    // Focus the input
    $('#new-chat-name').focus().select();
    
    // Handle close button
    $('.close-modal, .cancel-rename').on('click', function() {
        $('.rename-modal-overlay').remove();
    });
    
    // Handle save button
    $('.save-rename').on('click', function() {
        const newName = $('#new-chat-name').val().trim();
        if (newName && newName !== currentName) {
            // Clean and format the new name
            let title = newName;
            
            // Remove any special characters and limit length
            title = title.replace(/[^\w\s]/g, '');
            
            // If title is too long, truncate it and add ellipsis
            if (title.length > 30) {
                title = title.substring(0, 30) + "...";
            }
            
            // If title is empty after cleaning, use default
            if (!title) {
                title = "New Chat";
            }
            
            session.name = title;
            
            // Update the session name in the database
            $.ajax({
                url: '/api/update_session_name/',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken()
                },
                data: JSON.stringify({
                    session_id: sessionId,
                    name: title
                }),
                success: function(response) {
                    console.log('Session renamed:', response);
                    // Store the title in localStorage
                    localStorage.setItem(`chat_title_${sessionId}`, title);
                    updateSessionList();
                    $('.rename-modal-overlay').remove();
                },
                error: function(xhr, status, error) {
                    console.error('Error renaming session:', error);
                    alert('Failed to rename chat. Please try again.');
                }
            });
        } else {
            $('.rename-modal-overlay').remove();
        }
    });
    
    // Handle Enter key
    $('#new-chat-name').on('keypress', function(e) {
        if (e.which === 13) {
            $('.save-rename').click();
        } else if (e.which === 27) {
            $('.rename-modal-overlay').remove();
        }
    });
    
    // Handle click outside modal
    $('.rename-modal-overlay').on('click', function(e) {
        if ($(e.target).hasClass('rename-modal-overlay')) {
            $(this).remove();
        }
    });
  }

  // Add click handlers for session actions
  $(document).on('click', '.session-item', function(e) {
    if (!$(e.target).closest('.session-actions').length) {
        const sessionId = $(this).data('session-id');
        switchSession(sessionId);
    }
  });

  $(document).on('click', '.delete-session', function(e) {
    e.stopPropagation();
    const sessionId = $(this).data('session-id');
    deleteSession(sessionId);
  });

  $(document).on('click', '.rename-session', function(e) {
    e.stopPropagation();
    const sessionId = $(this).data('session-id');
    renameSession(sessionId);
  });

  // Function to switch sessions
  function switchSession(sessionId) {
    console.log('Switching to session:', sessionId);
    
    if (currentSessionId === sessionId) return;
    
    currentSessionId = sessionId;
    $("#message-list").empty();
    
    // Find the current session in our sessions array
    const currentSession = sessions.find(s => s.id === sessionId);
    if (!currentSession) {
        console.error('Session not found:', sessionId);
        return;
    }
    
    // Load chat history for the new session
    $.ajax({
        url: '/api/get_chat_history/',
        method: 'GET',
        data: { session_id: sessionId },
        success: function(messages) {
            console.log('Loaded messages:', messages);
            
            // Track unique messages to prevent duplicates
            const uniqueMessages = new Set();
            
            // Only load messages if there are any and they're not system messages
            if (messages && messages.length > 0) {
                messages.forEach(function(msg) {
                    // Skip system messages and empty messages
                    if (msg.role === 'system' || !msg.message.trim()) {
                        return;
                    }
                    
                    const isUser = msg.role === 'human';
                    // Remove "Question:" prefix from user messages
                    const messageText = isUser ? msg.message.replace(/^Question:\s*/i, '') : msg.message;
                    
                    // Create a unique key for the message
                    const messageKey = `${isUser ? 'user' : 'bot'}-${messageText}-${msg.timestamp}`;
                    
                    // Only append if we haven't seen this message before
                    if (!uniqueMessages.has(messageKey)) {
                        uniqueMessages.add(messageKey);
                        appendMessage(messageText, isUser, msg.timestamp);
                    }
                });
            }
            
            // Update the session list to reflect the current session
            updateSessionList();
            scrollToBottom();
        },
        error: function(xhr) {
            console.error('Error loading chat history:', xhr);
        }
    });
  }

  // Initialize on page load
  $(document).ready(function() {
    // Add the clear all history button after the new chat button
    $("#new-chat-btn").after(`
      <button id="clear-all-history-btn" class="btn btn-danger">
        <i class="fa fa-trash"></i> Clear All History
      </button>
    `);
    
    // Add click handler for clear all history button
    $("#clear-all-history-btn").on("click", clearAllChatHistory);
    
    // Remove any existing click handlers
    $("#new-chat-btn").off('click');
    
    // Add click handler for New Chat button
    $("#new-chat-btn").on("click", function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Check if session creation is locked
        if (sessionLock.isLocked) {
            console.log('Session creation is locked');
            return;
        }
        
        const $button = $(this);
        $button.prop('disabled', true);
        
        // Clear any ongoing operations
        if (reader) {
            reader.cancel();
            reader = null;
        }
        cancelRequested = true;
        
        // Clear the message list
        $("#message-list").empty();
        
        // Create new session
        createNewSession()
            .then(sessionId => {
                console.log('New session created:', sessionId);
                switchSession(sessionId);
                updatePDFStatus();
            })
            .catch(error => {
                console.error('Error creating new chat:', error);
                if (error.message !== 'Session creation is locked') {
                    alert('Failed to create new chat. Please try again.');
                }
            })
            .finally(() => {
                setTimeout(() => {
                    $button.prop('disabled', false);
                }, 1000);
            });
    });
    
    // Load initial sessions
    fetchSessionsFromDB().then(() => {
        if (sessions.length === 0) {
            createNewSession().then(sessionId => {
                switchSession(sessionId);
            });
        } else {
            switchSession(sessions[0].id);
        }
    });
  });

  // Update the clear chat functionality
  $("#clear-chat-btn").on("click", function () {
    if (!confirm("Are you sure you want to clear the current chat? This will only clear the current session.")) return;

    $.ajax({
      url: "/api/clear_chat_history/",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCSRFToken(),
      },
      data: JSON.stringify({ session_id: currentSessionId }),
      success: function(response) {
        $("#message-list").empty();
        updatePDFStatus();
        
        // Create a new session
        createNewSession().then(sessionId => {
          switchSession(sessionId);
        });
        
        alert("Current chat has been cleared.");
      },
      error: function(xhr, status, error) {
        console.error("Error clearing chat:", error);
        alert("❌ Failed to clear chat history. Please try again.");
      }
    });
  });

  function getCSRFToken() {
    return $('input[name="csrfmiddlewaretoken"]').val();
  }

  function showLoading() {
    $("#loading-animation").show();
  }

  function hideLoading() {
    $("#loading-animation").hide();
  }

  function appendMessage(content, isUser, timestamp = new Date().toISOString()) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.dataset.timestamp = timestamp;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    if (!isUser) {
        // Process markdown and formatting for bot messages
        const formattedContent = formatBotMessage(content);
        messageContent.innerHTML = formattedContent;
    } else {
        messageContent.textContent = content;
    }
    
    const messageTimestamp = document.createElement('div');
    messageTimestamp.className = 'message-timestamp';
    
    // Convert timestamp to local time
    const date = new Date(timestamp);
    messageTimestamp.textContent = date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        hour12: true
    });
    
    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTimestamp);
    chatWindow.append(messageDiv);
    scrollToBottom();
    return messageDiv.dataset.timestamp;
  }

  function formatBotMessage(content) {
    // Configure marked options
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                try {
                    return hljs.highlight(code, { language: lang }).value;
                } catch (e) {
                    console.error(e);
                }
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false,
        sanitize: false,
        tables: true,
        smartLists: true,
        smartypants: true
    });

    // First, check if this is a medical response with numbered sections
    if (content.match(/^\d\./)) {
        return formatMedicalResponse(content);
    }
    
    // Check for table content
    if (content.includes('Table Content:')) {
        return formatTableContent(content);
    }
    
    // Check for image content
    if (content.includes('Image Content:')) {
        return formatImageContent(content);
    }
    
    // Convert the content to HTML using marked
    let formattedContent = marked.parse(content);
    
    // Sanitize the HTML using DOMPurify with enhanced security
    formattedContent = DOMPurify.sanitize(formattedContent, {
        ALLOWED_TAGS: [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'br', 'strong', 'em', 'u', 's',
            'ul', 'ol', 'li', 'blockquote', 'pre', 'code', 'a', 'img', 'div', 'span',
            'table', 'thead', 'tbody', 'tr', 'th', 'td', 'caption', 'colgroup', 'col'
        ],
        ALLOWED_ATTR: [
            'href', 'src', 'alt', 'class', 'id', 'target', 'rel', 'style',
            'colspan', 'rowspan', 'scope', 'align', 'valign', 'width', 'height'
        ],
        FORBID_TAGS: ['script', 'style', 'iframe', 'object', 'embed'],
        FORBID_ATTR: ['onerror', 'onload', 'onclick', 'onmouseover']
    });
    
    // Add custom classes and enhance formatting
    formattedContent = formattedContent
        // Headers
        .replace(/<h([1-6])>/g, '<h$1 class="section-header">')
        .replace(/<h1>/g, '<h1 class="main-title">')
        .replace(/<h2>/g, '<h2 class="section-title">')
        .replace(/<h3>/g, '<h3 class="subsection-title">')
        // Lists
        .replace(/<ul>/g, '<ul class="custom-list">')
        .replace(/<ol>/g, '<ol class="custom-list">')
        .replace(/<li>/g, '<li class="list-item">')
        // Code blocks
        .replace(/<pre>/g, '<pre class="code-block">')
        .replace(/<code>/g, '<code class="code-inline">')
        // Links
        .replace(/<a href=/g, '<a class="custom-link" href=')
        // Tables
        .replace(/<table>/g, '<table class="custom-table">')
        .replace(/<thead>/g, '<thead class="table-header">')
        .replace(/<tbody>/g, '<tbody class="table-body">')
        .replace(/<tr>/g, '<tr class="table-row">')
        .replace(/<th>/g, '<th class="table-header-cell">')
        .replace(/<td>/g, '<td class="table-cell">');
    
    return formattedContent;
  }

  function formatTableContent(content) {
    // Extract table content
    const tableMatch = content.match(/Table Content:\n([\s\S]*?)(?=\n\n|$)/);
    if (!tableMatch) return content;
    
    const tableContent = tableMatch[1];
    
    // Create a formatted table
    const formattedTable = `
        <div class="table-container">
            <div class="table-header">Table Content</div>
            <div class="table-content">
                ${tableContent}
            </div>
        </div>
    `;
    
    return formattedTable;
  }

  function formatImageContent(content) {
    // Extract image content
    const imageMatch = content.match(/Image Content:\n([\s\S]*?)(?=\n\n|$)/);
    if (!imageMatch) return content;
    
    const imageContent = imageMatch[1];
    
    // Create a formatted image section
    const formattedImage = `
        <div class="image-container">
            <div class="image-header">Image Content</div>
            <div class="image-content">
                ${imageContent}
            </div>
        </div>
    `;
    
    return formattedImage;
  }

  function formatMedicalResponse(content) {
    // Split content into sections
    const sections = content.split(/\n(?=\d\.)/);
    
    // Format each section
    const formattedSections = sections.map(section => {
        // Extract section number and content
        const match = section.match(/^(\d+)\.\s*(.*)/s);
        if (!match) return section;
        
        const [_, number, content] = match;
        
        // Format the section
        return `
            <div class="medical-section">
                <div class="section-number">${number}</div>
                <div class="section-content">
                    ${marked.parse(content.trim())}
                </div>
            </div>
        `;
    });
    
    return formattedSections.join('');
  }

  // Remove the CSS styles section and fix the PDF status HTML
  function updatePDFStatus() {
    const storedFileName = localStorage.getItem(`pdf_file_name_${currentSessionId}`);
    if (storedFileName) {
      $("#pdf-status").html(`Current file: <strong>${storedFileName}</strong> <button id="remove-pdf" class="btn btn-sm btn-danger">Remove</button>`);
    } else {
      $("#pdf-status").html("No file selected");
    }
  }

  function updateMessage(msgId, newContent) {
    const messageDiv = document.querySelector(`[data-timestamp="${msgId}"]`);
    if (messageDiv) {
        const messageContent = messageDiv.querySelector('.message-content');
        if (messageContent) {
            if (messageDiv.classList.contains('bot-message')) {
                messageContent.innerHTML = formatBotMessage(newContent);
            } else {
                messageContent.textContent = newContent;
            }
        }
    }
    scrollToBottom();
  }

  function scrollToBottom() {
    if (!scrollLocked) {
      $("#chat-window").scrollTop($("#chat-window")[0].scrollHeight);
    }
  }

  $("#chat-window").on("scroll", function () {
    const scrollTop = $(this).scrollTop();
    const scrollHeight = this.scrollHeight;
    const clientHeight = $(this).innerHeight();
    scrollLocked = scrollTop + clientHeight < scrollHeight - 100;
  });

  // Initialize tooltip for paper-clip button
  paperClipBtn.tooltip({
    placement: 'top',
    trigger: 'hover',
    title: 'Upload files (PDF, Images, Documents)'
  });

  // Handle paper-clip button click
  paperClipBtn.on('click', function() {
    fileInput.click();
  });

  // Handle file selection
  fileInput.on('change', function(e) {
    const files = [...e.target.files];
    files.forEach(file => {
      if (isValidFileType(file)) {
        // Show loading message with dots
        const loadingMsgId = appendMessage(`⏳ Processing ${file.name}...`, false);
        let dotCount = 0;
        const dotsInterval = setInterval(() => {
          dotCount = (dotCount + 1) % 4;
          const dots = '.'.repeat(dotCount) || '.';
          updateMessage(loadingMsgId, `⏳ Processing ${file.name}${dots}`);
        }, 400);

        // Process the file immediately
        const formData = new FormData();
        formData.append('files', file);
        formData.append('session_id', currentSessionId);

        fetch('/api/upload_file/', {
          method: 'POST',
          body: formData,
          headers: {
            'X-CSRFToken': getCSRFToken()
          }
        })
        .then(response => response.json())
        .then(data => {
          // Clear the loading animation
          clearInterval(dotsInterval);
          
          if (data.error) {
            updateMessage(loadingMsgId, `❌ Error processing file: ${data.error}`);
          } else {
            // Store the file name for future queries
            if (data.files && data.files.length > 0) {
              const fileName = data.files[0].file_name;
              localStorage.setItem(`pdf_file_name_${currentSessionId}`, fileName);
              updatePDFStatus();
              
              // Add file information to chat history
              $.ajax({
                url: '/api/add_pdf_to_history/',
                method: 'POST',
                headers: {
                  'Content-Type': 'application/json',
                  'X-CSRFToken': getCSRFToken()
                },
                data: JSON.stringify({
                  session_id: currentSessionId,
                  message: `📄 File uploaded: ${fileName}`,
                  role: 'ai'
                }),
                success: function(response) {
                  if (response.timestamp) {
                    // Use the server's timestamp for the message
                    const messageDiv = document.querySelector(`[data-timestamp="${loadingMsgId}"]`);
                    if (messageDiv) {
                      messageDiv.dataset.timestamp = response.timestamp;
                      const timestampDiv = messageDiv.querySelector('.message-timestamp');
                      if (timestampDiv) {
                        const date = new Date(response.timestamp);
                        timestampDiv.textContent = date.toLocaleTimeString('en-US', {
                          hour: '2-digit',
                          minute: '2-digit',
                          hour12: true
                        });
                      }
                    }
                    updateMessage(loadingMsgId, `📄 File uploaded: ${fileName}`);
                  } else {
                    updateMessage(loadingMsgId, `📄 File uploaded: ${fileName}`);
                  }
                }
              });
            }
          }
        })
        .catch(error => {
          // Clear the loading animation
          clearInterval(dotsInterval);
          console.error('Error processing file:', error);
          updateMessage(loadingMsgId, `❌ Error processing file: ${error.message}`);
        });
      } else {
        alert(`Invalid file type. Please upload a PDF or image file (PNG, JPEG, GIF, BMP).`);
      }
    });
  });

  function isValidFileType(file) {
    const validTypes = {
        'application/pdf': 'pdf',
        'image/png': 'image',
        'image/jpeg': 'image',
        'image/jpg': 'image',
        'image/gif': 'image',
        'image/bmp': 'image'
    };
    
    // Check file type
    if (!validTypes[file.type]) {
        alert('Invalid file type. Please upload a PDF or image file (PNG, JPEG, GIF, BMP).');
        return false;
    }
    
    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB in bytes
    if (file.size > maxSize) {
        alert('File is too large. Maximum file size is 10MB.');
        return false;
    }
    
    return true;
  }

  // Handle send button click
  $("#send-button").on("click", function () {
    const userMessage = $("#message-input").val().trim();
    if (!userMessage) {
        alert("Please type a question.");
        return;
    }

    // Just send the message since files are already processed
    sendMessage(userMessage);
  });

  function sendMessage(userMessage) {
    showLoading();
    // Remove "Question:" prefix before displaying the message
    const displayMessage = userMessage.replace(/^Question:\s*/i, '');
    
    // Display user message immediately
    appendMessage(displayMessage, true);
    $("#message-input").val("");
    cancelRequested = false;

    // Update session name with the user's query if this is the first message
    const session = sessions.find(s => s.id === currentSessionId);
    if (session && (!session.name || session.name === "New Chat")) {
        updateSessionName(currentSessionId, displayMessage);
    }

    const requestData = {
        query: userMessage,
        session_id: currentSessionId,
        stream: true
    };

    const storedFileName = localStorage.getItem(`pdf_file_name_${currentSessionId}`);
    if (storedFileName && storedFileName.trim()) {
        requestData.file_name = storedFileName;
    }

    const controller = new AbortController();
    const signal = controller.signal;

    reader = null;
    let retryCount = 0;
    const maxRetries = 3;
    let fullMessage = "";
    let msgId = null;
    let isFirstChunk = true;
    let buffer = "";

    function startStreaming() {
        fetch("/api/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCSRFToken(),
            },
            body: JSON.stringify(requestData),
            signal: signal,
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            if (!response.body) {
                throw new Error("No response body.");
            }
            
            reader = response.body.getReader();
            const decoder = new TextDecoder();
            let lastChunkTime = Date.now();
            const chunkTimeout = 10000; // 10 seconds timeout

            function processChunk(chunk) {
                buffer += chunk;
                
                // Try to process complete JSON objects
                try {
                    const jsonChunk = JSON.parse(buffer);
                    if (jsonChunk.content) {
                        if (isFirstChunk) {
                            hideLoading();
                            msgId = appendMessage("🤖 " + jsonChunk.content, false);
                            isFirstChunk = false;
                            fullMessage = jsonChunk.content;
                        } else {
                            fullMessage += jsonChunk.content;
                            updateMessage(msgId, "🤖 " + fullMessage);
                        }
                        buffer = ""; // Clear buffer after successful processing
                    }
                } catch (e) {
                    // If not JSON, treat as plain text
                    if (isFirstChunk) {
                        hideLoading();
                        msgId = appendMessage("🤖 " + buffer, false);
                        isFirstChunk = false;
                        fullMessage = buffer;
                    } else {
                        fullMessage += buffer;
                        updateMessage(msgId, "🤖 " + fullMessage);
                    }
                    buffer = ""; // Clear buffer after processing
                }
            }

            function checkChunkTimeout() {
                const currentTime = Date.now();
                if (currentTime - lastChunkTime > chunkTimeout) {
                    throw new Error("Stream timeout - no data received");
                }
            }

            const readChunk = () => {
                return reader.read().then(({ done, value }) => {
                    if (done || cancelRequested) {
                        hideLoading();
                        if (fullMessage) {
                            updateMessage(msgId, "🤖 " + fullMessage);
                        } else if (buffer) {
                            // Process any remaining buffer content
                            processChunk(buffer);
                        } else {
                            updateMessage(msgId, "🤖 (no response)");
                        }
                        return;
                    }

                    lastChunkTime = Date.now();
                    const chunk = decoder.decode(value, { stream: true });
                    processChunk(chunk);
                    return readChunk();
                }).catch((error) => {
                    if (retryCount < maxRetries && !cancelRequested) {
                        retryCount++;
                        console.log(`Retrying stream (${retryCount}/${maxRetries})...`);
                        hideLoading();
                        setTimeout(startStreaming, 1000); // Wait 1 second before retry
                    } else {
                        throw error;
                    }
                });
            };

            // Start periodic timeout check
            const timeoutInterval = setInterval(checkChunkTimeout, 1000);

            return readChunk().finally(() => {
                clearInterval(timeoutInterval);
            });
        })
        .catch((err) => {
            hideLoading();
            if (cancelRequested) {
                appendMessage("❌ Response canceled by user.", false);
            } else {
                console.error("Stream error:", err);
                if (retryCount >= maxRetries) {
                    appendMessage("❌ Failed to get response after multiple attempts. Please try again.", false);
                } else {
                    appendMessage("❌ Error in response. Retrying...", false);
                }
            }
        });
    }

    startStreaming();
  }

  // Handle Enter key press in message input
  $("#message-input").on("keypress", function (e) {
    if (e.which === 13) {
      e.preventDefault();
      $("#send-button").click();
    }
  });

  // Handle PDF removal
  $(document).on("click", "#remove-pdf", function() {
    localStorage.removeItem(`pdf_file_name_${currentSessionId}`);
    updatePDFStatus();
    appendMessage("PDF removed. You can now ask general questions.", false);
  });

  function formatTimestamp(timestamp) {
    if (!timestamp) return '';
    try {
        // If timestamp is already in the correct format (YYYY-MM-DD HH:MM AM/PM)
        if (timestamp.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2} [AP]M$/)) {
            return timestamp;
        }
        
        // Parse the timestamp and convert to local time
        const date = new Date(timestamp);
        if (isNaN(date.getTime())) {
            console.error('Invalid timestamp:', timestamp);
            return '';
        }
        
        // Format the date in local timezone
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        }).replace(',', '');
    } catch (e) {
        console.error('Error formatting timestamp:', e, 'Timestamp:', timestamp);
        return '';
    }
  }

  // Function to clear all chat history
  function clearAllChatHistory() {
    if (!confirm("Are you sure you want to clear all chat history? This cannot be undone.")) {
      return;
    }

    // Clear all sessions from database
    $.ajax({
      url: '/api/clear_chat_history/',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken()
      },
      data: JSON.stringify({ clear_all: true }),
      success: function(response) {
        // Clear the sessions array
        sessions = [];
        currentSessionId = null;
        
        // Clear the session list in the UI
        sessionList.empty();
        
        // Clear the message list
        $("#message-list").empty();
        
        // Create a new session
        createNewSession().then(sessionId => {
          switchSession(sessionId);
        }).catch(error => {
          console.error('Error creating new chat:', error);
          alert('Failed to create new chat. Please try again.');
        });
        
        // Update PDF status
        updatePDFStatus();
        
        // Show success message
        appendMessage("All chat history has been cleared.", false);
      },
      error: function(xhr, status, error) {
        console.error('Error clearing chat history:', error);
        alert("Failed to clear chat history. Please try again.");
      }
    });
  }
});