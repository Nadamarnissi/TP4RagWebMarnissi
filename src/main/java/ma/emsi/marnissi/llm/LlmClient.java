
package ma.emsi.marnissi.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import jakarta.enterprise.context.Dependent;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Dependent
public class LlmClient implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private String systemRole;

    // Les objets LangChain4j ne sont pas sérialisés avec la vue JSF.
    private transient Assistant assistant;
    private transient ChatMemory chatMemory;

    public LlmClient() {
        initialiserAssistant();
    }

    /**
     * Initialisation du service IA avec mémoire et RAG.
     * Chaque instance de LlmClient possède sa propre conversation.
     */
    private void initialiserAssistant() {

        // Même propriété JVM que dans le TP2.
        String apiKey = System.getProperty("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException(
                    "La propriété JVM GEMINI_KEY est absente."
            );
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        // Mémoire propre à chaque conversation.
        chatMemory = MessageWindowChatMemory.withMaxMessages(20);

        // Deux récupérateurs construits une seule fois pour l'application.
        Map<ContentRetriever, String> sources =
                new LinkedHashMap<>();

        sources.put(
                RagResources.PDF_RAG,
                "Support de cours sur l'intelligence artificielle, " +
                        "les modèles de langage, les embeddings, " +
                        "le fine-tuning et le RAG."
        );

        sources.put(
                RagResources.PDF_ASTRONOMIE,
                "Document sur l'astronomie et le système solaire, " +
                        "notamment les planètes, les caractéristiques " +
                        "du système solaire et les objets astronomiques."
        );

        // Fonctionnalité 1 : reformuler les questions contextuelles.
        CompressingQueryTransformer queryTransformer =
                new CompressingQueryTransformer(model);

        // Fonctionnalité 2 : choisir le PDF pertinent avec Gemini.
        QueryRouter queryRouter =
                new LanguageModelQueryRouter(model, sources);

        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .queryRouter(queryRouter)
                        .build();

        assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();
    }

    /**
     * Préserve le comportement du TP2 :
     * un changement de rôle réinitialise la conversation.
     */
    public void setSystemRole(String systemRole) {

        this.systemRole = systemRole;

        chatMemory.clear();

        if (systemRole != null && !systemRole.isBlank()) {
            chatMemory.add(SystemMessage.from(systemRole));
        }
    }

    public String chat(String question) {

        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException(
                    "La question ne peut pas être vide."
            );
        }

        return assistant.chat(question);
    }

    /**
     * Reconstruit les ressources transitoires après
     * une éventuelle désérialisation de la vue JSF.
     */
    @Serial
    private void readObject(ObjectInputStream input)
            throws IOException, ClassNotFoundException {

        input.defaultReadObject();

        initialiserAssistant();

        if (systemRole != null && !systemRole.isBlank()) {
            chatMemory.add(SystemMessage.from(systemRole));
        }
    }

    /**
     * Ressources RAG partagées entre les conversations.
     *
     * Les PDF sont indexés une seule fois par classloader,
     * plutôt qu'à chaque création du backing bean JSF.
     */
    private static final class RagResources {

        private static final EmbeddingModel EMBEDDING_MODEL =
                new BgeSmallEnV15EmbeddingModel();

        private static final ContentRetriever PDF_RAG =
                creerRetriever("rag.pdf");

        private static final ContentRetriever PDF_ASTRONOMIE =
                creerRetriever("autre.pdf");

        private static ContentRetriever creerRetriever(String fichier) {

            DocumentParser parser = new ApacheTikaDocumentParser();

            Document document =
                    ClassPathDocumentLoader.loadDocument(
                            fichier,
                            parser
                    );

            DocumentSplitter splitter =
                    DocumentSplitters.recursive(300, 30);

            List<TextSegment> segments =
                    splitter.split(document);

            List<Embedding> embeddings =
                    EMBEDDING_MODEL.embedAll(segments).content();

            EmbeddingStore<TextSegment> store =
                    new InMemoryEmbeddingStore<>();

            store.addAll(embeddings, segments);

            System.out.println(
                    "[RAG] Document indexé : " + fichier +
                            " (" + segments.size() + " segments)"
            );

            return EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(store)
                    .embeddingModel(EMBEDDING_MODEL)
                    .maxResults(5)
                    .minScore(0.4)
                    .build();
        }

        private RagResources() {
        }
    }
}
