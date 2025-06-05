
  $(document).ready(function () {
    // Capture the Send button click event
    $('#send-button').on('click', function () {
      // Get the user's message
      const userMessage = $('#message-input').val();
      
      if (userMessage.trim() === "") {
        alert("Please enter a message");
        return;
      }

      // Display user message in chat window
      const userMsgHtml = `<div class='message-line my-text'><div class='message-box my-text'>${userMessage}</div></div>`;
      $('#message-list').append(userMsgHtml);
      
      // Clear input field after sending
      $('#message-input').val('');

      // Prepare the request data
      const requestData = {
        query: userMessage
      };

      // Send the message to Flask API
      $.ajax({
        url: '/chatbot',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function (response) {
          // Display chatbot response in chat window
          const botMsgHtml = `<div class='message-line'><div class='message-box'>${response.response}</div></div>`;
          $('#message-list').append(botMsgHtml);

          // Scroll to the bottom of chat window
          $('#chat-window').scrollTop($('#chat-window')[0].scrollHeight);
        },
        error: function (error) {
          console.error("Error in sending message:", error);
          alert("An error occurred while processing your request.");
        }
      });
    });

    // Optionally: Press "Enter" to send a message
    $('#message-input').on('keypress', function (e) {
      if (e.which == 13) { // Enter key
        $('#send-button').click();
      }
    });
  });
