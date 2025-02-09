// === IDL-CREATION-GUIDLINES === // Object Oriented: Use OO Design. // Design Patterns: Use Factory, Builder and Strategy patterns where possible // ** Complex parameters JSON : Use JSON where primitive params are not possible and document them in IDL like "Expected JSON format: { "key1": "type1", "key2": "type2" }" // == !! BEGIN IDL TEMPLATE !! === // === CODE-CREATION-RULES === // Strict Typing: Always use strict typing. Avoid using ambiguous or variant types. // Primitive Types: Favor the use of primitive types wherever possible. // Portability Mandate: Python code must be written with the intent to be ported to Java, Go, and JavaScript. Consider language-agnostic logic and avoid platform-specific dependencies. // No Side Effects: Functions should be pure, meaning their output should only be determined by their input without any observable side effects. // Testability: Ensure that every function and method is easily testable. Avoid tight coupling and consider dependency injection where applicable. // Documentation: Every function, method, and module should be thoroughly documented, especially if there's a nuance that's not directly evident from its signature. // Contractual Obligation: The definitions provided in this IDL are a strict contract. All specified interfaces, methods, and constraints must be implemented precisely as defined without deviation. // =======================

module GenericSystemName {

// Interface for a generic entity
interface EntityName {

    // Action/method definition
    // Preconditions:
    // - Define any preconditions here.
    // - Expected JSON format: { "key1": "type1", "key2": "type2" } 
    // Postconditions:
    // - Define the expected outcomes here.
    returnType methodName(parameterType parameterName);

    // Additional methods...
};

// Another entity or component
interface AnotherEntity {

    // Action/method definition
    // Preconditions:
    // - Define any preconditions here.
    // - Expected JSON format: { "key1": "type1", "key2": "type2" } 
    // Postconditions:
    // - Define the expected outcomes here.
    returnType anotherMethodName(parameterType parameterName);

    // Additional methods...
};
// == !! END IDL TEMPLATE !! ===

// EXAMPLE // === CODE-CREATION-RULES === // Strict Typing: Always use strict typing. Avoid using ambiguous or variant types. // Primitive Types: Favor the use of primitive types wherever possible. // Portability Mandate: Python code must be written with the intent to be ported to Java, Go, and JavaScript. Consider language-agnostic logic and avoid platform-specific dependencies. // No Side Effects: Functions should be pure, meaning their output should only be determined by their input without any observable side effects. // Testability: Ensure that every function and method is easily testable. Avoid tight coupling and consider dependency injection where applicable. // Documentation: Every function, method, and module should be thoroughly documented, especially if there's a nuance that's not directly evident from its signature. // Contractual Obligation: The definitions provided in this IDL are a strict contract. All specified interfaces, methods, and constraints must be implemented precisely as defined without deviation. // ======================= // == !! BEGIN TEMPLATE EXAMPLE !! === interface Tweets { // Preconditions: // - userID exists. // - tweetContent is non-null and within allowable size limits. // Postconditions: // - A new tweet is created and stored. // Expected JSON format: { "userID": "string", "content": "string" } void postTweet(string tweetJSON); // Preconditions: // - userID and tweetID exist. // Postconditions: // - The tweet with tweetID is marked as liked by userID. void likeTweet(string userID, string tweetID); // Preconditions: // - userID and tweetID exist. // Postconditions: // - The tweet with tweetID is marked as retweeted by userID. // Expected JSON format: { "userID": "string", "originalTweetID": "string" } void retweet(string retweetJSON); // Preconditions: // - tweetID exists. // Postconditions: // - Returns the details of the tweet as JSON. string getTweetDetails(string tweetID); // Invariants: // - Filesystem storage maintains a list of tweets, likes, and retweets for each tweetID. }; // == !! END TEMPLATE EXAMPLE !! ===
)
